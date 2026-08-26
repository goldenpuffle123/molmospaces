"""Reference client for the external-module bridge -- COPY THIS FILE.

This is the whole integration surface. It imports numpy, msgpack-numpy and
websockets and NOTHING from ``molmo_spaces``, on purpose: an external stack
should not need this benchmark installed to be driven by it, and a test in
``mlspaces_tests/component_tests/test_bridge.py`` pins that property.

Implement three methods and serve them:

    class MyStack:
        def reset(self, episode):        # build your planner from the metadata
            return {"subscribe": ["head_camera_depth", "sensor_param_head_camera"]}
        def step(self, obs, info):       # one decision per control period
            return {"base": [x, y, theta], "done": False}
        def close(self, final):          # {"episode_id", "progress"}; return anything to record
            return {"frames": self.n}

    serve(MyStack(), port=8800)

``serve`` greets each client with ``{"protocol": PROTOCOL, "stack": ...}`` the
moment it connects; the sim reads that frame before sending anything and refuses
a version it does not speak. If you write your own transport, send it too.

``role="module"`` instead returns an observation AUGMENTATION from ``step`` --
a dict merged into the observation before the policy sees it. That is how a
perception stack goes in FRONT of a planner without either of them importing
the other.

CONVENTIONS you get in ``obs`` (see ``bridge/protocol.py`` for the full list):

* ``{cam}_depth``            metric z-depth [m], float32, NOT ray length
* ``{cam}_segmentation``     [H, W, 3]; channel 2 is the BODY id
* ``{cam}_self_mask``        True where the pixel is the robot's own body
* ``camera_params_{cam}``    ``cam2world_cv`` (OpenCV frame) + ``intrinsic_cv``
* ``base_pose_mat``          4x4 base->world, MEASURED
* ``contacts``               [N, 6] world position + normal, robot vs world
"""

from __future__ import annotations

import traceback
from typing import Any

import msgpack_numpy
import numpy as np
import websockets.exceptions
import websockets.sync.server

PROTOCOL = 1


class ScenePriorView:
    """``episode["prior"]`` with the two accessors mapping stacks want.

    Mirrors ``molmo_spaces.env.scene_prior.ScenePrior`` without importing it.
    """

    def __init__(self, wire: dict) -> None:
        self.wire = wire
        self.privilege = wire.get("privilege")
        self.withheld_categories = list(wire.get("withheld_categories") or [])
        self.semantics = list(wire.get("semantics") or [])
        self.geometry = wire.get("geometry")

    @property
    def geometry_points(self) -> np.ndarray:
        g = self.geometry
        return np.zeros((0, 3), np.float32) if not g else np.asarray(g["points"], np.float32)

    def vocabulary(self) -> list[str]:
        return sorted({str(o["class_name"]) for o in self.semantics})

    def as_point_clouds(self) -> list[dict]:
        return [
            {"class_name": str(o["class_name"]), "points": np.asarray(o["points"], np.float32)}
            for o in self.semantics
            if len(o["points"])
        ]

    def summary(self) -> str:
        return (
            f"ScenePrior(privilege={self.privilege}, {len(self.semantics)} objects, "
            f"{len(self.geometry_points)} geometry points, "
            f"withheld {len(self.withheld_categories)} categories)"
        )


def handle_message(handler: Any, msg: dict) -> dict:
    """One request -> one reply. Transport-free, so it is unit-testable."""
    op = msg.get("op")
    if msg.get("protocol") != PROTOCOL:
        return {"error": f"protocol {msg.get('protocol')} != {PROTOCOL}"}
    role = msg.get("role", "policy")
    if op == "reset":
        out = handler.reset(msg["episode"]) or {}
        return {"ok": True, **out}
    if op == "step":
        info = {"step": msg.get("step", 0), "t": msg.get("t", 0.0), "progress": msg.get("progress")}
        out = handler.step(msg["obs"], info)
        key = "action" if role == "policy" else "obs_augment"
        if isinstance(out, dict) and (key in out or "info" in out):
            return out
        return {key: out, "info": {}}
    if op == "close":
        # The id matters: episodes can overlap on a long-lived connection (see
        # protocol.close_message), so a handler must be told WHICH one ended.
        final = {"episode_id": msg.get("episode_id"), "progress": msg.get("progress")}
        out = getattr(handler, "close", lambda _f: None)(final) or {}
        return {"ok": True, "info": out}
    return {"error": f"unknown op {op!r}"}


def serve(handler: Any, host: str = "127.0.0.1", port: int = 8800, verbose: bool = True) -> None:
    """Serve one connection at a time, forever.

    Sequential by design: the sim drives one episode at a time over one
    long-lived connection, and a stack holding a GPU model wants exactly one
    copy of itself. An exception is answered with an ``error`` reply carrying
    the traceback -- the sim side then raises loudly instead of stalling, which
    is the failure mode a socket integration otherwise hides.
    """

    def _conn(ws):
        if verbose:
            print(f"[bridge] connected: {ws.remote_address}", flush=True)
        # GREET FIRST. The sim's transport waits for a metadata frame on connect
        # (the same helper its learned-policy client uses), and the version in it
        # is what turns "this stack speaks an older protocol" into one clear line
        # instead of a missing key three messages later.
        ws.send(msgpack_numpy.packb(
            {"protocol": PROTOCOL, "stack": type(handler).__name__}))
        try:
            for raw in ws:
                msg = msgpack_numpy.unpackb(raw)
                try:
                    reply = handle_message(handler, msg)
                except Exception:
                    reply = {"error": traceback.format_exc()}
                    print(
                        f"[bridge] handler raised on {msg.get('op')!r}:\n{reply['error']}",
                        flush=True,
                    )
                ws.send(msgpack_numpy.packb(reply))
        except websockets.exceptions.ConnectionClosed:
            # The sim exiting drops the socket without a close frame. That is a
            # NORMAL end of run, not an error: swallow it and go back to
            # listening, so the next run reuses this process (and whatever it
            # spent 10-50 s loading) instead of a traceback and a dead port.
            pass
        if verbose:
            print("[bridge] client disconnected; listening again", flush=True)

    with websockets.sync.server.serve(
        _conn, host, port, compression=None, max_size=None, ping_interval=None
    ) as server:
        if verbose:
            print(f"[bridge] listening on ws://{host}:{port}", flush=True)
        server.serve_forever()
