"""``BridgePolicy``: run an episode from ANOTHER PROCESS, in any environment.

RELATIONSHIP TO WHAT IS ALREADY HERE, stated first because it is most of the
design. ``policy/learned_policy/websocket_policy.py`` already forwards an
observation to an external process over msgpack-numpy + websockets and applies
the action it returns -- SAME transport, SAME direction. This is not a new pipe.
What it adds is the part a PLANNER needs and a VLA does not:

* a typed ``reset`` carrying the episode (robot, cameras, scene bounds, action
  space, target category, prior) -- ``WebsocketPolicy`` sends pixels, qpos and a
  task string, and treats a socket reconnect as the episode boundary;
* an action that is not hardwired to ``{"arm", "gripper"}``;
* proprioception no sensor exposes (contacts, actuator commands, the measured
  footprint), which in-process policies read off ``MjModel``/``MjData``;
* a privilege-stamped progress channel, and a ``close`` that returns per-episode
  info for the record;
* a ROLE tag, so the same transport also serves a perception module.

Both ``WebsocketPolicy`` and ``WebsocketPolicyServer`` currently have no callers
in this tree (the Pi path uses ``openpi_client``), so this is a candidate to
subsume them rather than sit beside them. The TRANSPORT is already shared:
``ws_connect`` / ``ws_request`` / ``ws_uri`` live in ``websocket_policy.py`` and
both clients call them, so there is one place where "msgpack-numpy to a model
process" is implemented and one set of conventions (server greets, string reply
means error, refusals are retried).

    MolmoSpaces process                 external process (any env, any language)
    -------------------                 ----------------------------------------
    BridgePolicy.reset()   --reset-->   build your planner from the metadata
    BridgePolicy.get_action --step-->   one control decision
    BridgePolicy.close()   --close-->   flush, return episode info

WHY IT IS NOT POLICY-ONLY. The protocol is ROLE-TAGGED, so the same transport
serves a chain:

    sim --obs--> [module] --obs + augmentation--> [policy] --action--> sim

A perception module returns an observation AUGMENTATION instead of an action.
That is what an out-of-process detector / semantic mapper is, and without the
role the integration is a bespoke socket in someone else's repo that nobody
else can reuse.

WHY OUT-OF-PROCESS IS THE POINT. An in-process integration makes the external
stack share this venv, i.e. share our pins -- and this tree carries a commit
titled "Pin warp version to avoid incompatibility" for exactly that reason. The
first two clients of this bridge pin a different mujoco AND a different warp;
one of them needs torch-nightly against our torch~=2.7. A bridge makes that
class of conflict structurally impossible instead of negotiable.

IN-PROCESS STAYS FIRST-CLASS. ``BridgePolicy`` is *a* ``BasePolicy``, not a
replacement for one. A policy that wants ``MjData`` should not pay
serialisation for it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from molmo_spaces.bridge import protocol as P
from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.policy.learned_policy.websocket_policy import ws_connect, ws_request, ws_uri

logger = logging.getLogger(__name__)

# uri -> connection, ONE PER PROCESS and deliberately not per episode. The
# pipeline builds a fresh policy object for every episode and closes it in
# cleanup_episode_resources; a per-episode socket would ask the far side to
# rebuild whatever it loaded (a perception stack measured 10-51 s per reload)
# and, against a single-connection server, can deadlock the run outright -- the
# new connection sits in the listen backlog while the old one is never read to
# EOF. Episode boundaries are the "reset" and "close" OPS on one long-lived
# connection instead.
_CONNECTIONS: dict[str, Any] = {}

# Episode ids have to be unique across POLICY OBJECTS, not within one: the
# pipeline builds a policy per episode, so a per-object counter would label
# every episode "-0" and a client keyed on the id could not tell them apart.
_EPISODE_COUNTER = [0]


def _next_episode_id() -> int:
    _EPISODE_COUNTER[0] += 1
    return _EPISODE_COUNTER[0]


# uri -> the greeting the endpoint sent when it connected.
_GREETINGS: dict[str, dict] = {}


def _connect(uri: str, timeout: float | None) -> Any:
    """Open (or reuse) the connection to one endpoint, and check its version.

    The socket work is ``websocket_policy.ws_connect`` -- the same helper the
    learned-policy client uses, so there is one implementation of "talk
    msgpack-numpy to a model process" in the tree rather than two. What is
    bridge-specific is above it: the connection is CACHED for the process (see
    ``_CONNECTIONS``) and the greeting is checked against this protocol version,
    which turns a mismatched client into one clear line instead of a KeyError
    somewhere in an episode payload.
    """
    conn = _CONNECTIONS.get(uri)
    if conn is not None:
        return conn
    try:
        conn, greeting = ws_connect(uri, connection_timeout=timeout)
    except (TimeoutError, RuntimeError) as e:
        raise P.BridgeError(f"no bridge endpoint at {uri}: {e}") from e
    P.check_greeting(greeting, uri)
    _CONNECTIONS[uri], _GREETINGS[uri] = conn, greeting
    logger.info("bridge endpoint %s: %s", uri, greeting)
    return conn


def _to_jsonable(x):
    """numpy -> plain python, so a remote ``info`` dict can be written out."""
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


class _Endpoint:
    """One websocket peer, with the role it plays in the chain."""

    def __init__(self, uri: str, role: str, timeout: float | None) -> None:
        if role not in P.ROLES:
            raise P.BridgeError(f"role must be one of {P.ROLES}, got {role!r}")
        self.uri, self.role = ws_uri(uri), role
        self.timeout = timeout
        self.subscribe: list[str] | None = None
        self.meta: dict = {}
        self.info: dict = {}

    def send(self, msg: dict) -> dict:
        conn = _connect(self.uri, self.timeout)
        try:
            reply = ws_request(conn, msg)
        except Exception as e:  # a dropped peer must not look like a bad plan
            _CONNECTIONS.pop(self.uri, None)
            raise P.BridgeError(f"bridge transport to {self.uri} failed: {e!r}") from e
        return P.check_reply(reply, msg["op"], self.role)

    @property
    def greeting(self) -> dict:
        return _GREETINGS.get(self.uri, {})


class BridgePolicy(BasePolicy):
    """Forward the episode to external endpoints; return their action.

    Knobs live in ``BridgePolicyConfig`` (``configs/bridge_configs.py``).
    """

    def __init__(self, config, task=None) -> None:
        super().__init__(config, task)
        pc = config.policy_config
        self.pc = pc
        self.endpoints = [_Endpoint(u, "module", pc.connection_timeout) for u in pc.module_uris]
        self.policy_ep = _Endpoint(pc.policy_uri, "policy", pc.connection_timeout)
        self.endpoints.append(self.policy_ep)
        self.step_i = 0
        self.episode_i = 0
        self._episode_id = ""
        self._sent_reset = False
        self._oracle = None
        self._last_action: dict | None = None
        # (model id -> robot body ids). Scanning every body by name is a few ms
        # on a house-sized model and the answer only changes with the scene, so
        # it must not sit in the per-step path.
        self._bodies_cache: tuple[int, np.ndarray] | None = None

    # ---- sensors -------------------------------------------------------- #
    def create_policy_sensors(self):
        """The perception bundle the external side asked for, in config.

        Rendering is a CONFIG decision, not a protocol one: sensors are built
        before any episode exists, so the far side cannot choose them in
        ``reset``. What it CAN choose there is the subscription -- which of the
        rendered streams are actually put on the wire.
        """
        from molmo_spaces.env.sensors import get_perception_sensors

        base = super().create_policy_sensors()
        if not self.pc.cameras:
            return base
        return base + get_perception_sensors(
            self.config,
            cameras=list(self.pc.cameras),
            depth=self.pc.depth,
            segmentation=self.pc.segmentation,
            self_mask=self.pc.self_mask,
            include_params=self.pc.camera_params,
        )

    # ---- episode -------------------------------------------------------- #
    def _env(self):
        return self.task.env

    def _index(self) -> int:
        return self._env().current_batch_index

    def _base_pose(self) -> np.ndarray:
        return np.asarray(self._env().current_robot.robot_view.base.pose, float)

    def _robot_bodies(self) -> np.ndarray:
        m = self._env().current_model
        if self._bodies_cache is None or self._bodies_cache[0] != id(m):
            self._bodies_cache = (id(m), P.robot_body_ids(m, self.config.robot_config.robot_namespace))
        return self._bodies_cache[1]

    def _episode(self) -> dict:
        """Everything the far side gets before the first frame."""
        from molmo_spaces.env.task_oracle import Privilege, TaskOracle

        env, task, cfg = self._env(), self.task, self.config
        m, d = env.current_model, env.current_data
        ns = cfg.robot_config.robot_namespace
        bodies = self._robot_bodies()
        priv = Privilege(self.pc.oracle_privilege)
        self._oracle = TaskOracle(task, priv, dt_s=self.pc.dt_s)

        cams = []
        for spec in cfg.camera_config.cameras:
            if self.pc.cameras and spec.name not in self.pc.cameras:
                continue
            reg = env.camera_manager.registry
            cam = reg[spec.name] if spec.name in reg else None
            cams.append(
                {
                    "name": spec.name,
                    "fov": float(cam.fov) if cam is not None else None,
                    "resolution": list(cfg.camera_config.img_resolution),
                    "record_depth": bool(getattr(spec, "record_depth", False)),
                    # Measured at reset, not declared: mounts are noised per
                    # episode (RBY1's head tilt by 0.2 rad), so a client that
                    # trusted the config would plan a gaze it cannot execute.
                    "pos": np.asarray(cam.pos, float).tolist() if cam is not None else None,
                    "forward": (
                        np.asarray(cam.forward, float).tolist() if cam is not None else None
                    ),
                    "up": np.asarray(cam.up, float).tolist() if cam is not None else None,
                }
            )

        # nav_objs is the nav task's candidate list; other task types simply have
        # no candidates and the oracle block comes back empty rather than raising.
        all_objs = getattr(task, "nav_objs", None) or []
        objs = list(all_objs[self._index()]) if self._index() < len(all_objs) else []
        category = getattr(task.config.task_config, "pickup_obj_category", None)
        if not category and objs:
            category = str(objs[0].name).split("_")[0]
        instruction = ""
        try:
            instruction = str(task.get_task_description())
        except Exception:
            pass
        natural = ""
        try:
            om = env.object_managers[self._index()]
            natural = str(om.fallback_expression(objs[0].name)).lower()
        except Exception:
            pass

        lo, hi = P.scene_aabb(m, d)
        ep = {
            "protocol": P.PROTOCOL,
            "episode_id": f"{getattr(self.config, 'tag', 'episode')}-{_next_episode_id()}",
            "task_type": str(getattr(cfg, "task_type", "")),
            "instruction": instruction,
            "target_category": category,
            "target_natural_name": natural,
            "horizon_steps": int(getattr(task, "_task_horizon", 0) or 0),
            "policy_dt_s": float(self.pc.dt_s),
            "seed": getattr(cfg, "seed", None),
            "robot": {
                "name": getattr(cfg.robot_config, "name", None),
                "namespace": ns,
                "command_modes": dict(getattr(cfg.robot_config, "command_mode", {}) or {}),
                # The ROBOT decides this, not the policy config: a nonholonomic
                # platform driven through "holo_joint_planar_position" can be
                # commanded to strafe, and a planner told "holonomic" would do
                # exactly that. BridgePolicyConfig.base_kinematics is an
                # override for the deliberate case and defaults to None.
                "base_kinematics": (
                    self.pc.base_kinematics
                    or getattr(cfg.robot_config, "base_kinematics", "holonomic")
                ),
                "base_pose": self._base_pose(),
                "footprint": P.robot_footprint(m, d, self._base_pose(), bodies),
                "body_ids": bodies,
            },
            "cameras": cams,
            "scene": {
                "bounds": [lo, hi],
                # P0b's contract, as a FIELD: a consumer of segmentation ids can
                # assert it instead of trusting that some sampler set it.
                "offsamples": int(m.vis.quality.offsamples),
                "model_path": str(getattr(env, "current_model_path", "")),
            },
            "action_space": {
                "base": {"shape": [3], "kind": self.pc.base_action_kind},
                "done": {"kind": "bool"},
            },
            # Opaque, config-declared knobs for the far side. The benchmark-side
            # config stays the ONE definition site for an experiment's arms even
            # though the planner runs elsewhere.
            "params": dict(self.pc.params),
            "prior": P.scene_prior_to_wire(self._build_prior(env, category, bodies)),
            # PRIVILEGED, and stamped so a recording says what was handed over.
            "oracle": {
                "privilege": priv.value,
                "candidates": [
                    {
                        "name": o.name,
                        "body_ids": P.descendant_bodies(m, m.body(o.name).id),
                        # GT poses are SHAPING-and-up: a simulated detector needs
                        # the identity of its target (body ids), not its answer.
                        "position": (
                            np.asarray(o.position, float).tolist()
                            if priv != Privilege.UNPRIVILEGED
                            else None
                        ),
                    }
                    for o in objs
                ],
            },
            "extra": dict(self.pc.episode_extra),
        }
        return ep

    def _build_prior(self, env, category, bodies):
        if self.pc.prior_geometry == "none" and self.pc.prior_semantics == "none":
            return None
        from molmo_spaces.env.scene_prior import export_scene_prior

        withhold = set(self.pc.prior_withhold_categories)
        if self.pc.prior_withhold_target and category:
            withhold.add(category)
        return export_scene_prior(
            env,
            geometry=self.pc.prior_geometry,
            semantics=self.pc.prior_semantics,
            withhold_categories=withhold,
            withhold_movable=self.pc.prior_withhold_movable,
            voxel=self.pc.prior_voxel,
            skip_bodies={int(b) for b in bodies},
        )

    def reset(self) -> None:
        ep = self._episode()
        self._episode_id = ep["episode_id"]
        self.step_i = 0
        self._last_action = None
        for e in self.endpoints:
            reply = e.send(P.reset_message(e.role, ep))
            sub = reply.get("subscribe")
            e.subscribe = None if sub is None else [str(s) for s in sub]
            e.meta = dict(reply.get("meta") or {})
        self.episode_i += 1
        self._sent_reset = True

    # ---- per step ------------------------------------------------------- #
    def _obs(self, observation) -> dict:
        o = observation[0] if isinstance(observation, (list, tuple)) else observation
        m, d = self._env().current_model, self._env().current_data
        ns = self.config.robot_config.robot_namespace
        bodies = set(int(b) for b in self._robot_bodies())
        rv = self._env().current_robot.robot_view
        out = dict(o)
        # NOT "robot_base_pose": the nav task ships a sensor by that name holding
        # the same fact in a DIFFERENT encoding (7D position+quaternion).
        # Overwriting it would silently change what an existing consumer reads.
        out["base_pose_mat"] = self._base_pose()
        try:
            out["base_qpos"] = np.asarray(rv.get_move_group("base").joint_pos, float)
        except Exception:
            out["base_qpos"] = None
        out["contacts"] = P.world_contacts(m, d, bodies)
        out["robot_ctrl"] = P.robot_ctrl(m, d, ns)
        return out

    def _progress(self, force: bool = False) -> dict | None:
        """``TaskOracle`` at the declared tier, on the declared cadence.

        Cadence matters: the JUDGE tier renders a segmentation frame to answer
        "is it visible", so asking every step adds a render per step. 0 = never.
        ``force`` is the end of the episode, where the verdict is wanted once
        regardless of where the cadence happened to land.

        PHASE: the payload goes out on the LAST step of each window (steps
        N-1, 2N-1, ...), not the first. A client counts its own frames from 1,
        so this is what puts a fresh verdict on ITS frame N -- the alternative
        hands it a payload from N-1 steps ago and every diagnostic printed
        beside it silently disagrees with the trajectory it sits next to.
        """
        every = int(self.pc.progress_every)
        if not every or self._oracle is None or (not force and (self.step_i + 1) % every):
            return None
        try:
            return self._oracle.progress()
        except Exception as e:  # a diagnostic must never end an episode
            return {"error": f"{type(e).__name__}: {e}"}

    def get_action(self, observation):
        if not self._sent_reset:
            self.reset()
        obs = self._obs(observation)
        progress = self._progress()
        t = self.step_i * float(self.pc.dt_s)

        augmented: set[str] = set()
        for e in self.endpoints[:-1]:  # modules first, in order
            msg = P.step_message(e.role, self.step_i, t, self._filter(obs, e, augmented), progress)
            reply = e.send(msg)
            aug = reply.get("obs_augment") or {}
            obs.update(aug)
            # A module's output is never filtered away by the next endpoint's
            # subscription: the subscription names SENSORS, and an augmentation
            # is the reason the module is in the chain at all.
            augmented |= set(aug)
            e.info = dict(reply.get("info") or {})

        e = self.policy_ep
        reply = e.send(
            P.step_message(e.role, self.step_i, t, self._filter(obs, e, augmented), progress)
        )
        e.info = dict(reply.get("info") or {})
        self.step_i += 1
        action = reply["action"]
        if not isinstance(action, dict):
            raise P.BridgeError(f"action must be a dict, got {type(action).__name__}")
        action = {k: (np.asarray(v, float) if isinstance(v, (list, tuple)) else v)
                  for k, v in action.items()}
        self._last_action = action
        return action

    def _filter(self, obs: dict, e: _Endpoint, augmented: set[str] = frozenset()) -> dict:
        """Apply the endpoint's ``reset`` subscription.

        640x480 RGB + float32 depth + uint32 segmentation is ~3.7 MB a frame;
        sending streams the far side throws away is the one avoidable cost in
        this design, so declaring a subscription is worth the four lines.
        """
        if e.subscribe is None:
            return obs
        keep = (
            set(e.subscribe)
            | set(augmented)
            | {"base_pose_mat", "base_qpos", "contacts", "robot_ctrl"}
        )
        return {k: v for k, v in obs.items() if k in keep}

    # ---- teardown ------------------------------------------------------- #
    def get_info(self) -> dict:
        info = dict(super().get_info())
        info["bridge_protocol"] = P.PROTOCOL
        info["bridge_policy_uri"] = self.policy_ep.uri
        # WHICH stack produced these numbers, from its own greeting. A recording
        # that cannot name the far side is not reproducible.
        info["bridge_peer"] = _to_jsonable(self.policy_ep.greeting)
        info["oracle_privilege"] = str(self.pc.oracle_privilege)
        info.update(_to_jsonable(self.policy_ep.info))
        for i, e in enumerate(self.endpoints[:-1]):
            if e.info:
                info[f"module_{i}_info"] = _to_jsonable(e.info)
        return info

    def close(self) -> None:
        """End the EPISODE on every endpoint; keep the connections.

        The pipeline closes the policy object per episode. Tearing the sockets
        down here would make the far side reload whatever it loaded, so the
        episode boundary is the ``close`` op and the connection outlives it.
        """
        if not self._sent_reset:
            return
        final = self._progress(force=True)
        for e in self.endpoints:
            try:
                reply = e.send(P.close_message(e.role, self._episode_id, final))
                e.info = dict(reply.get("info") or e.info)
            except P.BridgeError as err:
                logger.warning("bridge close failed for %s: %s", e.uri, err)
        self._sent_reset = False
