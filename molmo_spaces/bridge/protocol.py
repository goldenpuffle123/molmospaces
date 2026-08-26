"""Wire protocol for the external-module bridge.

``BridgePolicy`` (see ``policy.py``) speaks this to any process that can read
msgpack over a websocket. Three ops, one reply each:

    reset(episode) -> {"ok": true,  "subscribe": [...], "meta": {...}}
    step(obs)      -> {"action": {...}, "info": {...}}          role="policy"
                   -> {"obs_augment": {...}, "info": {...}}     role="module"
    close(final)   -> {"ok": true, "info": {...}}

WHAT THE PAYLOADS ARE FOR. The point of a bridge is not the socket, it is that
the OBSERVATION BECOMES THE CONTRACT: an in-process policy can read the model,
the renderer, the contacts and the task's own answer, and nobody can tell from
the outside which of those it used. Every field below exists because a real
client needed it and reaching past the protocol was the alternative -- the
footprint measurement, the contact list and the actuator commands are all
proprioception a deployed robot has, and they are named here rather than being
read off ``MjData`` behind the benchmark's back.

PRIVILEGE. Two blocks are privileged and both say so in the payload:
``episode["oracle"]`` (the simulated detector's target body ids, i.e. the
identity of the answer) and ``step["progress"]`` (``TaskOracle`` at the tier the
run declared). A run's recording therefore states what its policy was given.
``episode["params"]["_privilege"]`` is not a thing -- the stamp travels in the
payload the numbers came in, which is what makes it greppable.

VERSIONING. ``protocol`` is 1. Both directions carry an open ``info`` / ``meta``
dict so clients can experiment without a version bump; anything that changes the
MEANING of an existing field is a bump.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from molmo_spaces.utils.mj_model_and_data_utils import descendant_bodies as _descendant_bodies

PROTOCOL = 1
ROLES = ("policy", "module")


class BridgeError(RuntimeError):
    """A protocol violation, or an error the far side reported."""


# --------------------------------------------------------------------------- #
# small model reads the protocol needs (all cheap, all done once per episode)
# --------------------------------------------------------------------------- #
def robot_body_ids(model, namespace: str) -> np.ndarray:
    """Body ids under a robot's namespace."""
    return np.array(
        [
            b
            for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(namespace)
        ],
        dtype=np.int32,
    )


def descendant_bodies(model, body_id: int) -> list[int]:
    """``body_id`` and every body descended from it, as a sorted list.

    The segmentation frame's channel 2 carries the body id of the geom that was
    hit, so a consumer that wants "pixels belonging to this object" needs the
    body SET, not the single id -- otherwise a multi-body asset (a cabinet with
    a door) reports only the part whose body happens to own the geom. This is
    the id-space equivalent of ``get_geom_seg_mask``'s descendant-geom lookup,
    and it is what lets an external client compute the same mask without the
    model. The set comes from ``utils.mj_model_and_data_utils`` -- one
    definition, wire-shaped here.
    """
    return sorted(_descendant_bodies(model, int(body_id)))


def scene_aabb(model, data, margin: float = 0.0) -> tuple[list[float], list[float]]:
    """World bounding box of the compiled scene, from geom centres +- rbound.

    Map EXTENT is not privileged: the shipped A* baseline is handed the full
    occupancy grid, and this is only the box that grid would live in.
    """
    c = data.geom_xpos[: model.ngeom]
    r = model.geom_rbound[: model.ngeom, None]
    lo = np.min(c - r, axis=0) - margin
    hi = np.max(c + r, axis=0) + margin
    return lo.tolist(), hi.tolist()


def robot_footprint(model, data, base_pose: np.ndarray, bodies: np.ndarray) -> dict:
    """The robot's MEASURED extent in its own base frame.

    A planner needs the footprint it must keep clear, and a spec sheet is not
    it: a mobile manipulator drives with its arms at ``init_qpos`` and its torso
    above them, none of which is in a chassis rectangle. Measuring the compiled
    model instead is exact and tracks the pose the episode actually starts in.

    Each geom is taken as its BOUNDING SPHERE (centre +- max(geom_size)), which
    is conservative -- a long thin box contributes its longest half-side on both
    axes. That is the intended bias for a footprint a planner must keep clear.

    Returns half-extents (base frame), the z range the body occupies, and the
    same half-extents per z band -- the bands are what a body-height collision
    model needs, and computing them here means every client gets the same
    numbers instead of each one re-deriving them from ``geom_size``.
    """
    bx, by = float(base_pose[0, 3]), float(base_pose[1, 3])
    c, s = float(base_pose[0, 0]), float(base_pose[1, 0])  # cos/sin of base yaw
    ext = []
    for b in bodies:
        adr, num = int(model.body_geomadr[b]), int(model.body_geomnum[b])
        for g in range(adr, adr + num):
            gp = data.geom_xpos[g]
            r = float(np.max(model.geom_size[g])) or 0.0
            dx, dy = float(gp[0]) - bx, float(gp[1]) - by
            ext.append((c * dx + s * dy, -s * dx + c * dy, float(gp[2]), r))
    if not ext:
        return {"half_extents": [0.0, 0.0], "z_range": [0.0, 0.0], "bands": []}
    e = np.asarray(ext, float)
    bands = []
    for z0, z1 in ((0.15, 0.5), (0.5, 0.9), (0.9, 1.2), (1.2, 1.5)):
        m = (e[:, 2] >= z0) & (e[:, 2] < z1)
        if m.any():
            bands.append(
                [
                    z0,
                    z1,
                    float(np.max(np.abs(e[m, 0]) + e[m, 3])),
                    float(np.max(np.abs(e[m, 1]) + e[m, 3])),
                ]
            )
    return {
        "half_extents": [
            float(np.max(np.abs(e[:, 0]) + e[:, 3])),
            float(np.max(np.abs(e[:, 1]) + e[:, 3])),
        ],
        "z_range": [float(e[:, 2].min()), float(e[:, 2].max())],
        "bands": bands,
    }


def world_contacts(model, data, bodies: set[int]) -> np.ndarray:
    """Robot-vs-world contacts as ``[N, 6]`` (world position, contact normal).

    PROPRIOCEPTION, not scene knowledge: this is what a bumper or a wheel-current
    estimator reports. Robot-vs-robot contacts are dropped (self-collision is the
    robot's own problem, not an obstacle), and no normal filtering happens here --
    which contacts count as "pressing on a wall" is the client's policy.
    """
    out = []
    for k in range(data.ncon):
        con = data.contact[k]
        r1 = int(model.geom_bodyid[con.geom1]) in bodies
        r2 = int(model.geom_bodyid[con.geom2]) in bodies
        if r1 != r2:
            out.append([*np.asarray(con.pos, float), *np.asarray(con.frame, float)[:3]])
    return np.asarray(out, np.float64).reshape(-1, 6)


def robot_ctrl(model, data, namespace: str) -> dict[str, float]:
    """Commanded value of every actuator under the robot's namespace.

    "The command I sent" vs "the joint I measure" is the only way to tell a
    planner that never asked for motion from a servo that refused to deliver it,
    and a policy that cannot see it re-runs the episode to find out.
    """
    out = {}
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if name.startswith(namespace):
            out[name] = float(data.ctrl[a])
    return out


def scene_prior_to_wire(prior) -> dict | None:
    """``ScenePrior`` -> plain msgpack-able dict (see ``env/scene_prior.py``).

    Field names are kept identical so the wire form reads as the dataclass; the
    reference client (``bridge/client.py``) wraps it back into something with
    ``vocabulary()`` / ``as_point_clouds()``.
    """
    if prior is None:
        return None
    return {
        "protocol": int(prior.protocol),
        "privilege": str(prior.privilege),
        "geometry": (
            None
            if prior.geometry is None
            else {
                "kind": prior.geometry.kind,
                "voxel": float(prior.geometry.voxel),
                "points": np.asarray(prior.geometry.points, np.float32),
            }
        ),
        "semantics": [
            {
                "name": o.name,
                "class_name": o.class_name,
                "natural_name": o.natural_name,
                "position": np.asarray(o.position, np.float32),
                "points": np.asarray(o.points, np.float32),
                "is_structural": bool(o.is_structural),
                "is_receptacle": bool(o.is_receptacle),
                "movable": bool(o.movable),
            }
            for o in prior.semantics
        ],
        "withheld_categories": list(prior.withheld_categories),
        "withheld_bodies": [int(b) for b in prior.withheld_bodies],
        "provenance": dict(prior.provenance),
    }


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #
def greeting(stack: str = "", **extra) -> dict:
    """What an endpoint sends the instant a client connects.

    The transport (``websocket_policy.ws_connect``) already waits for a metadata
    frame -- ``WebsocketPolicyServer`` has always sent one -- so the version
    check is free: a client that speaks a different protocol says so before an
    episode payload is built, instead of failing on a missing key mid-run.
    """
    return {"protocol": PROTOCOL, "stack": stack, **extra}


def check_greeting(msg: Any, uri: str) -> dict:
    """Validate an endpoint's greeting, or raise naming both versions."""
    if not isinstance(msg, dict):
        raise BridgeError(
            f"endpoint at {uri} greeted with {type(msg).__name__}, expected a dict "
            f"(see bridge/client.py: serve() sends protocol.greeting() on connect)"
        )
    got = msg.get("protocol")
    if got != PROTOCOL:
        raise BridgeError(
            f"endpoint at {uri} speaks bridge protocol {got!r}, this sim speaks "
            f"{PROTOCOL}. Re-copy molmo_spaces/bridge/client.py into that stack."
        )
    return msg


def reset_message(role: str, episode: dict) -> dict:
    return {"op": "reset", "protocol": PROTOCOL, "role": role, "episode": episode}


def step_message(role: str, step: int, t: float, obs: dict, progress: dict | None) -> dict:
    return {
        "op": "step",
        "protocol": PROTOCOL,
        "role": role,
        "step": int(step),
        "t": float(t),
        "obs": obs,
        "progress": progress,
    }


def close_message(role: str, episode_id: str, progress: dict | None = None) -> dict:
    """End of one episode, NAMED.

    ``episode_id`` is load-bearing rather than decorative: the pipeline builds a
    fresh policy object per episode and closes the old one AFTER the new one has
    reset, so a long-lived connection sees reset(A), reset(B), close(A),
    step(B). A close that just meant "drop your state" would take episode B's
    planner down with it -- measured, as a step arriving at a client that had
    just been told to close. With the id, a client finalises the episode the
    message names and keeps the live one running.
    """
    return {
        "op": "close",
        "protocol": PROTOCOL,
        "role": role,
        "episode_id": episode_id,
        "progress": progress,
    }


def check_reply(reply: Any, op: str, role: str) -> dict:
    """Validate one reply, or raise ``BridgeError`` naming what was wrong.

    A bridge fails at exactly two places -- the far side is not there, or it
    answered something the caller cannot use -- and both have to be loud. A
    silently-ignored malformed reply becomes a policy that does nothing, which
    reads as a bad planner.
    """
    if isinstance(reply, str):
        raise BridgeError(f"remote {role} error on {op}:\n{reply}")
    if not isinstance(reply, dict):
        raise BridgeError(f"remote {role} returned {type(reply).__name__}, expected a dict")
    if reply.get("error"):
        raise BridgeError(f"remote {role} error on {op}: {reply['error']}")
    if op == "step":
        key = "action" if role == "policy" else "obs_augment"
        if key not in reply:
            raise BridgeError(
                f"remote {role} reply to step has no {key!r} (keys: {sorted(reply)}). "
                f"A role={role!r} endpoint must answer with {key!r}."
            )
    return reply
