"""The external-module bridge: protocol, chain, and the copyable client.

What these pin, in order of what would actually break:

* an out-of-process POLICY drives an episode (reset -> step -> close) and its
  action arrives as the action the pipeline would apply;
* a role="module" endpoint in FRONT of it augments the observation, and the
  augmentation survives the policy's own (narrow) subscription -- the chain is
  the reason the protocol is role-tagged at all;
* the payload carries the fields an external planner cannot otherwise get
  without reaching past the benchmark (measured footprint, contacts, actuator
  commands, scene bounds, the MSAA setting) and STAMPS the privileged ones;
* ``bridge/client.py`` imports nothing from ``molmo_spaces``. That file is the
  integration surface; if it needs this package installed, the bridge has not
  decoupled anything.
"""

import ast
import pathlib
import threading
import time

import msgpack_numpy
import mujoco
import numpy as np
import pytest

from molmo_spaces.bridge import client as bclient
from molmo_spaces.bridge import protocol as P
from molmo_spaces.bridge import policy as bpolicy
from molmo_spaces.bridge.policy import BridgePolicy

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot_0/base" pos="0 0 0.2">
      <joint name="robot_0/slide" type="slide" axis="1 0 0"/>
      <geom name="robot_0/chassis" type="box" size="0.45 0.35 0.2"/>
      <body name="robot_0/head" pos="0 0 1.2">
        <geom name="robot_0/head_geom" type="sphere" size="0.1"/>
      </body>
    </body>
    <body name="mug_0" pos="2 0 0.5">
      <geom name="mug_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="mug_0/handle" pos="0.06 0 0">
        <geom name="mug_handle_geom" type="sphere" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="robot_0/slide_act" joint="robot_0/slide"/></actuator>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# the smallest task/env that BridgePolicy can read
# --------------------------------------------------------------------------- #
class _Cam:
    name, fov = "head_camera", 139.0
    pos = np.array([0.0, 0.0, 1.4])
    forward = np.array([1.0, 0.0, -0.5])
    up = np.array([0.0, 0.0, 1.0])


class _Registry:
    """Mimics CameraRegistry's surface: `name in reg` and `reg[name]`, nothing else."""

    def __init__(self, cams):
        self._cams = cams

    def __contains__(self, k):
        return k in self._cams

    def __getitem__(self, k):
        return self._cams[k]


class _Obj:
    def __init__(self, name, position):
        self.name, self.position = name, np.asarray(position, float)


class _Base:
    def __init__(self):
        self.pose = np.eye(4)


class _MoveGroup:
    joint_pos = np.zeros(3)


class _RobotView:
    def __init__(self):
        self.base = _Base()

    def get_move_group(self, name):
        return _MoveGroup()


class _Env:
    current_batch_index = 0
    current_model_path = "stub.xml"

    def __init__(self, model, data):
        self.current_model, self.current_data = model, data
        self.current_robot = type("_R", (), {"robot_view": _RobotView()})()
        self.robots = [self.current_robot]
        # NOT a dict: the real CameraRegistry supports only `in` and `[]`, and a
        # dict stub let a `.get()` slip through to a live run once.
        self.camera_manager = type("_CM", (), {"registry": _Registry({"head_camera": _Cam()})})()
        self.object_managers = [type("_OM", (), {"fallback_expression": lambda s, n: "coffee mug"})()]

    def check_visibility(self, camera_name, name):
        return 0.02


class _Task:
    episode_step_count = 3
    _task_horizon = 500

    def __init__(self, model, data):
        self.env = _Env(model, data)
        self.nav_objs = [[_Obj("mug_0", (2.0, 0.0, 0.5))]]
        self.config = type(
            "_C",
            (),
            {"task_config": type("_T", (), {"succ_pos_threshold": 1.5, "pickup_obj_category": "mug"})()},
        )()

    def get_task_description(self):
        return "find the mug"

    def get_reward(self):
        return [0.0]

    def get_nearest_nav_object(self, index):
        return self.nav_objs[0][0]


def _config(robot_kinematics: str = "holonomic", **policy_kw):
    from molmo_spaces.configs.bridge_configs import BridgePolicyConfig

    pc = BridgePolicyConfig(**policy_kw)
    robot = type(
        "_RC",
        (),
        {
            "robot_namespace": "robot_0/",
            "command_mode": {"base": "position"},
            "name": "stub",
            "base_kinematics": robot_kinematics,
        },
    )()
    cams = type("_CC", (), {"cameras": [_Cam()], "img_resolution": (640, 480)})()
    return type(
        "_Cfg",
        (),
        {
            "policy_config": pc,
            "robot_config": robot,
            "camera_config": cams,
            "task_type": "nav_to_obj",
            "seed": 7,
            "tag": "test",
        },
    )()


class _Recorder:
    """A policy endpoint that records what it was given and drives straight."""

    def __init__(self):
        self.episode = None
        self.steps = []
        self.closed = None

    def reset(self, episode):
        self.episode = episode
        return {"subscribe": ["head_camera_depth"], "meta": {"stack": "test"}}

    def step(self, obs, info):
        self.steps.append((obs, info))
        return {"base": [1.0, 2.0, 0.5], "done": len(self.steps) >= 2}

    def close(self, final):
        self.closed = final
        return {"frames": len(self.steps)}


class _Tagger:
    """A module endpoint: returns an augmentation, never an action."""

    def reset(self, episode):
        return {"subscribe": ["head_camera_depth"]}

    def step(self, obs, info):
        return {"semantic_objects": [{"label": "mug", "score": 0.9}]}


@pytest.fixture()
def model_data():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _serve(handler, port):
    t = threading.Thread(target=bclient.serve, args=(handler,), kwargs={"port": port, "verbose": False}, daemon=True)
    t.start()
    for _ in range(100):  # the socket is up within a few ms; poll rather than sleep long
        try:
            import socket

            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return t
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("bridge test server did not come up")


# --------------------------------------------------------------------------- #
# transport-free protocol checks
# --------------------------------------------------------------------------- #
def test_handle_message_dispatch():
    rec = _Recorder()
    assert bclient.handle_message(rec, {"op": "reset", "protocol": 1, "role": "policy", "episode": {"a": 1}})["ok"]
    assert rec.episode == {"a": 1}
    reply = bclient.handle_message(
        rec, {"op": "step", "protocol": 1, "role": "policy", "step": 0, "t": 0.0, "obs": {}, "progress": None}
    )
    assert reply["action"]["base"] == [1.0, 2.0, 0.5]
    assert "protocol" in bclient.handle_message(rec, {"op": "step", "protocol": 99})["error"]
    assert "unknown op" in bclient.handle_message(rec, {"op": "fly", "protocol": 1})["error"]


def test_module_reply_key_is_role_specific():
    reply = bclient.handle_message(
        _Tagger(), {"op": "step", "protocol": 1, "role": "module", "step": 0, "t": 0.0, "obs": {}}
    )
    assert "obs_augment" in reply and "action" not in reply
    # ... and a module answering a policy's question is a loud error, not a hang.
    with pytest.raises(P.BridgeError):
        P.check_reply(reply, "step", "policy")


def test_descendant_bodies_covers_child_bodies(model_data):
    model, _ = model_data
    ids = P.descendant_bodies(model, model.body("mug_0").id)
    assert model.body("mug_0/handle").id in ids, "a multi-body asset must report all its bodies"


def test_footprint_is_measured_not_declared(model_data):
    model, data = model_data
    bodies = P.robot_body_ids(model, "robot_0/")
    fp = P.robot_footprint(model, data, np.eye(4), bodies)
    # Per-geom BOUNDING SPHERE, so the 0.45 x 0.35 x 0.2 chassis contributes
    # 0.45 on both axes -- deliberately conservative, see robot_footprint.
    assert fp["half_extents"] == pytest.approx([0.45, 0.45], abs=1e-6)
    assert fp["z_range"][1] == pytest.approx(1.4, abs=1e-6)
    assert any(b[0] == 1.2 for b in fp["bands"])


# --------------------------------------------------------------------------- #
# the whole loop, over a real socket
# --------------------------------------------------------------------------- #
def test_bridge_drives_an_episode(model_data):
    model, data = model_data
    rec = _Recorder()
    _serve(rec, 8931)
    policy = BridgePolicy(_config(policy_uri="ws://127.0.0.1:8931", progress_every=1), _Task(model, data))
    policy.reset()

    ep = rec.episode
    assert ep["protocol"] == P.PROTOCOL
    assert ep["target_category"] == "mug"
    assert ep["target_natural_name"] == "coffee mug"
    assert ep["cameras"][0]["fov"] == 139.0
    assert ep["scene"]["offsamples"] == int(model.vis.quality.offsamples)
    assert ep["robot"]["footprint"]["half_extents"][0] == pytest.approx(0.45, abs=1e-6)
    # privileged block, stamped
    assert ep["oracle"]["privilege"] == "judge"
    assert model.body("mug_0/handle").id in ep["oracle"]["candidates"][0]["body_ids"]

    obs = {"head_camera_depth": np.ones((8, 8), np.float32), "head_camera": np.zeros((8, 8, 3), np.uint8)}
    action = policy.get_action([obs])
    assert action["done"] is False
    assert np.allclose(action["base"], [1.0, 2.0, 0.5])

    sent, info = rec.steps[0]
    assert "head_camera_depth" in sent
    assert "head_camera" not in sent, "subscription must drop streams nobody asked for"
    assert sent["base_pose_mat"].shape == (4, 4)
    assert sent["contacts"].shape[1] == 6
    assert "robot_0/slide_act" in sent["robot_ctrl"]
    assert info["progress"]["privilege"] == "judge"  # progress_every=1: every step

    assert policy.get_action([obs])["done"] is True
    policy.close()
    assert rec.closed["progress"] is not None, "close must carry the final progress payload"
    assert rec.closed["episode_id"] == ep["episode_id"], "close must name the episode it ends"
    assert policy.get_info()["frames"] == 2


def test_progress_lands_on_the_last_step_of_its_window(model_data):
    model, data = model_data
    rec = _Recorder()
    _serve(rec, 8934)
    policy = BridgePolicy(_config(policy_uri="ws://127.0.0.1:8934", progress_every=2), _Task(model, data))
    policy.reset()
    obs = {"head_camera_depth": np.ones((4, 4), np.float32)}
    policy.get_action([obs])
    policy.get_action([obs])
    got = [s[1]["progress"] is not None for s in rec.steps]
    assert got == [False, True], (
        "a client counting frames from 1 must get its fresh verdict ON frame N, "
        f"not N-1 steps stale (got {got})")
    policy.close()


def test_module_chain_augments_the_observation(model_data):
    model, data = model_data
    rec = _Recorder()
    _serve(_Tagger(), 8932)
    _serve(rec, 8933)
    policy = BridgePolicy(
        _config(
            policy_uri="ws://127.0.0.1:8933",
            module_uris=["ws://127.0.0.1:8932"],
            progress_every=0,
        ),
        _Task(model, data),
    )
    policy.reset()
    policy.get_action([{"head_camera_depth": np.ones((4, 4), np.float32)}])
    sent, info = rec.steps[0]
    assert sent["semantic_objects"][0]["label"] == "mug", (
        "a module's augmentation must reach the policy even though the policy's "
        "subscription names only sensors"
    )
    assert info["progress"] is None, "progress_every=0 must send nothing"
    policy.close()


def test_overlapping_episodes_close_by_id(model_data):
    """reset(A), reset(B), close(A), step(B) -- the pipeline's real ordering.

    The policy object is rebuilt per episode and closed only after the next one
    has reset, so a close that meant "drop your state" would kill the live
    episode. Measured as an AttributeError on the client one step later.
    """
    model, data = model_data

    class _TwoEpisode:
        def __init__(self):
            self.live, self.closed, self.steps = None, [], 0

        def reset(self, episode):
            self.live = episode["episode_id"]
            return {}

        def step(self, obs, info):
            assert self.live is not None, "stepped after the live episode was closed"
            self.steps += 1
            return {"base": [0.0, 0.0, 0.0], "done": False}

        def close(self, final):
            self.closed.append(final["episode_id"])
            if final["episode_id"] == self.live:
                self.live = None
            return {}

    h = _TwoEpisode()
    _serve(h, 8935)
    task = _Task(model, data)
    a = BridgePolicy(_config(policy_uri="ws://127.0.0.1:8935", progress_every=0), task)
    a.reset()
    b = BridgePolicy(_config(policy_uri="ws://127.0.0.1:8935", progress_every=0), task)
    b.reset()
    a.close()  # the late close of the PREVIOUS episode
    b.get_action([{"head_camera_depth": np.ones((4, 4), np.float32)}])
    assert h.steps == 1 and h.live is not None
    assert len(set(h.closed)) == 1, "each episode id must be closed once"
    b.close()


def test_greeting_carries_the_protocol_version(model_data):
    """The transport waits for a metadata frame; the version rides in it.

    Sharing ws_connect with the learned-policy client means the greeting is
    free, and a stack with a stale copy of client.py has to say so before an
    episode is built rather than failing on a missing key mid-episode.
    """
    model, data = model_data
    _serve(_Recorder(), 8936)
    policy = BridgePolicy(_config(policy_uri="ws://127.0.0.1:8936", progress_every=0), _Task(model, data))
    policy.reset()
    assert policy.policy_ep.greeting["protocol"] == P.PROTOCOL
    assert policy.policy_ep.greeting["stack"] == "_Recorder"
    assert policy.get_info()["bridge_peer"]["stack"] == "_Recorder"
    policy.close()

    # ... and a mismatch is one line, not a KeyError somewhere downstream.
    with pytest.raises(P.BridgeError, match="protocol"):
        P.check_greeting({"protocol": 99}, "ws://x")
    with pytest.raises(P.BridgeError, match="greeted with"):
        P.check_greeting("hello", "ws://x")


def test_bridge_uses_the_shared_websocket_transport():
    """One implementation of "msgpack-numpy to a model process", not two.

    This is the anti-duplication check: the bridge must call the helpers in
    websocket_policy.py (which WebsocketPolicy itself calls) rather than
    re-open a websocket and re-pack msgpack of its own.
    """
    src = pathlib.Path(bpolicy.__file__).read_text()
    assert "from molmo_spaces.policy.learned_policy.websocket_policy import" in src
    for rebuilt in ("websockets.sync.client.connect", "msgpack_numpy.packb", "msgpack_numpy.unpackb"):
        assert rebuilt not in src, f"bridge/policy.py re-implements {rebuilt}"


def test_websocket_policy_still_works_on_the_shared_helpers():
    """The learned-policy client, after the transport was factored out of it.

    ``WebsocketPolicy`` had no test and no in-tree caller, so extracting its
    connect/request bodies was an unverified edit on shipped code. This drives
    the real class against a real socket: greet, connect, round trip.
    """
    import websockets.sync.server

    from molmo_spaces.policy.learned_policy.websocket_policy import WebsocketPolicy

    def _echo(ws):
        ws.send(msgpack_numpy.packb({"model_name": "stub", "checkpoint": "none"}))
        for raw in ws:
            req = msgpack_numpy.unpackb(raw)
            ws.send(msgpack_numpy.packb({"arm": np.zeros(7), "gripper": np.array([req["g"]])}))

    server = websockets.sync.server.serve(_echo, "127.0.0.1", 8937, compression=None, max_size=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cfg = type("_C", (), {"policy_config": type("_P", (), {"force_enable_depth": False})()})()
    policy = WebsocketPolicy(cfg, model_name="stub", host="127.0.0.1", port=8937)
    policy.prepare_model()
    assert policy.get_server_metadata()["model_name"] == "stub", "the greeting must still be read"
    out = policy.infer({"g": 0.5})
    assert out["gripper"][0] == 0.5 and out["arm"].shape == (7,)
    action = policy.model_output_to_action(out)
    assert set(action) == {"arm", "gripper"}
    policy.close()
    server.shutdown()


def test_reference_client_does_not_import_molmo_spaces():
    src = pathlib.Path(bclient.__file__).read_text()
    tree = ast.parse(src)
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    assert not [m for m in mods if m.startswith("molmo_spaces")], (
        f"bridge/client.py is the copyable integration surface; it imports {mods}"
    )


def test_base_kinematics_comes_from_the_robot(model_data):
    """The ROBOT decides its motion model; the policy config only overrides it.

    This is the field that tells an external planner whether it may strafe. It is
    declared on ``BaseRobotConfig`` (Stretch sets "differential"), and a bridge
    that answered from its own default would tell every planner "holonomic" --
    the sim executes the strafe, the real drivetrain cannot, and nothing in the
    recording says so.
    """
    model, data = model_data

    # 1. Default: whatever the robot says, differential included.
    rec = _Recorder()
    _serve(rec, 8937)
    p = BridgePolicy(
        _config(robot_kinematics="differential", policy_uri="ws://127.0.0.1:8937"),
        _Task(model, data),
    )
    p.reset()
    assert rec.episode["robot"]["base_kinematics"] == "differential"
    p.close()

    # 2. An explicit config value still wins, for the deliberate case.
    rec2 = _Recorder()
    _serve(rec2, 8938)
    p2 = BridgePolicy(
        _config(
            robot_kinematics="differential",
            policy_uri="ws://127.0.0.1:8938",
            base_kinematics="holonomic",
        ),
        _Task(model, data),
    )
    p2.reset()
    assert rec2.episode["robot"]["base_kinematics"] == "holonomic"
    p2.close()
