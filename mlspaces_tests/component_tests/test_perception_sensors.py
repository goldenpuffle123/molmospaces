"""The perception bundle: posed depth (+segmentation, +self-mask) for mappers.

`get_nav_task_sensors` attaches RGB and camera parameters only, so a mapping
stack driving a nav task has to bypass the observation dict entirely and call
`env.render_depth_frame` itself. `get_perception_sensors` is the composable
bundle that closes that, and these tests pin its contract.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors import get_perception_sensors
from molmo_spaces.env.sensors_cameras import (
    CameraParameterSensor,
    CameraSensor,
    DepthSensor,
    SegmentationSensor,
    SelfOcclusionMaskSensor,
)

H, W = 48, 64


class _Cam:
    def __init__(self, name):
        self.name = name


class _CamCfg:
    img_resolution = (W, H)  # (width, height), as MolmoSpaces stores it

    def __init__(self, names):
        self.cameras = [_Cam(n) for n in names]


class _ExpCfg:
    def __init__(self, names=("head_camera", "wrist_camera_l")):
        self.camera_config = _CamCfg(names)


class _Env:
    """Minimal env exposing only what these sensors touch."""

    def __init__(self, seg):
        self._seg = seg
        self.calls = []

    def render_segmentation_frame(self, camera_name):
        self.calls.append(camera_name)
        return self._seg


def _seg_frame(body_ids):
    seg = np.zeros((H, W, 3), dtype=np.int32)
    seg[..., 2] = body_ids
    return seg


def test_bundle_covers_every_camera_by_default():
    s = get_perception_sensors(_ExpCfg(), depth=True, segmentation=True, self_mask=True)
    uuids = {x.uuid for x in s}
    for cam in ("head_camera", "wrist_camera_l"):
        assert {
            f"sensor_param_{cam}",
            f"{cam}_depth",
            f"{cam}_segmentation",
            f"{cam}_self_mask",
        } <= uuids


def test_streams_are_opt_in():
    s = get_perception_sensors(_ExpCfg(("head_camera",)), depth=True)
    assert {x.uuid for x in s} == {"sensor_param_head_camera", "head_camera_depth"}


def test_include_params_false_composes_without_uuid_collision():
    """The nav bundle already provides sensor_param_*; SensorSuite requires
    unique uuids, so composing must be possible without duplicating them."""
    cfg = _ExpCfg(("head_camera",))
    existing = get_perception_sensors(cfg, depth=False)  # params only
    extra = get_perception_sensors(cfg, depth=True, include_params=False)
    SensorSuite(existing + extra)  # must not raise
    assert all(not x.uuid.startswith("sensor_param") for x in extra)


def test_unknown_camera_fails_loud():
    with pytest.raises(AssertionError, match="unknown camera"):
        get_perception_sensors(_ExpCfg(), cameras=["nose_camera"])


def test_segmentation_sensor_renders_per_camera():
    """Regression: SegmentationSensor used to call env.segmentation_frame as a
    method. It is a property backed by _segmentation_frame, which is set to None
    and never assigned, so both branches failed and every observation was zeros.
    """
    seg = _seg_frame(np.full((H, W), 7))
    env = _Env(seg)
    out = SegmentationSensor(camera_name="head_camera", img_resolution=(W, H)).get_observation(
        env, task=None
    )
    assert env.calls == ["head_camera"], "must render the sensor's OWN camera"
    np.testing.assert_array_equal(out, seg)
    assert out.any(), "segmentation must not be all zeros"


def test_self_mask_flags_robot_bodies_only():
    ids = np.zeros((H, W), dtype=np.int32)
    ids[:10, :] = 3  # a robot body
    ids[10:20, :] = 9  # world geometry
    env = _Env(_seg_frame(ids))

    sensor = SelfOcclusionMaskSensor(
        camera_name="head_camera", img_resolution=(W, H), robot_namespace="robot_0/"
    )
    sensor._robot_body_ids = lambda _env: np.array([3], dtype=np.int32)

    mask = sensor.get_observation(env, task=None)
    assert mask.dtype == bool and mask.shape == (H, W)
    assert mask[:10, :].all(), "robot pixels must be flagged"
    assert not mask[10:, :].any(), "world pixels must not be flagged"


# --------------------------------------------------------------------------- #
# UUID naming, and the docs that describe it
#
# The names below are the whole integration surface for an external stack: an
# observation dict is keyed by uuid, and BridgePolicy._filter() drops anything a
# subscription does not name. An unrecognised name is IGNORED, not rejected, so a
# misspelling costs a silently empty stream that looks exactly like a broken
# sensor -- there is no loud failure anywhere on that path.
#
# The tests above already pinned what the CODE produces, and the docs drifted
# anyway: until 2026-08-28 this function's own docstring, bridge/client.py (the
# file external stacks copy verbatim), docs/bridge.md and GUIDE.md all told
# clients to read `camera_params_{cam}`, which no production bundle produces --
# every caller passes `sensor_param_` explicitly. So the drift guard has to be
# against the prose, not only the behaviour.
# --------------------------------------------------------------------------- #

_REPO = Path(__file__).resolve().parents[2]

# What get_perception_sensors actually builds, as templates over the camera name.
CANONICAL_UUIDS = {
    "sensor_param_{cam}",
    "{cam}_depth",
    "{cam}_segmentation",
    "{cam}_self_mask",
}

# Any uuid-shaped token in prose. Catches both affix orders, which is the point:
# `{cam}_depth` and `depth_{cam}` are equally plausible-looking and only one is real.
_UUID_TOKEN = re.compile(r"\{cam\}_[a-z_]+|[a-z_]+_\{cam\}")


def _documented_uuids(text: str) -> set[str]:
    """Every uuid-shaped token in prose. No exceptions and no negation handling:
    a doc that needs to warn about a wrong spelling names the PREFIX
    (``camera_params_``) rather than a whole uuid, so nothing legitimate is
    uuid-shaped and the guard needs no heuristic to tell warnings from claims.

    That is deliberate. The first version of this test allowed a bogus token when
    a negation appeared within N characters before it, and it MISSED the real bug
    in bridge/client.py -- whose previous line ended "NOT ray length", a negation
    about something else entirely. Proximity cannot distinguish those; not
    writing the token can.
    """
    flat = " ".join(text.replace("*", "").replace("`", "").split())
    return {m.group(0) for m in _UUID_TOKEN.finditer(flat)}


def test_bare_sensor_defaults_match_what_the_bundles_pass():
    """ONE uuid scheme, so a bare construction and a bundled one agree.

    These classes used to default to a second, PREFIXED scheme
    (`camera_params_{cam}`, `depth_{cam}`, ...) that no caller ever selected,
    because every bundle passes `uuid=` explicitly. Nothing consumed the
    defaults, so the disagreement was invisible in behaviour and surfaced only in
    the docs -- which described the defaults while clients had to use the bundle
    names. Pinning them equal is what stops that from recurring.
    """
    cam = "head_camera"
    assert CameraSensor(camera_name=cam, img_resolution=(W, H)).uuid == cam
    assert DepthSensor(camera_name=cam, img_resolution=(W, H)).uuid == f"{cam}_depth"
    assert (
        SegmentationSensor(camera_name=cam, img_resolution=(W, H)).uuid == f"{cam}_segmentation"
    )
    assert (
        SelfOcclusionMaskSensor(camera_name=cam, img_resolution=(W, H)).uuid == f"{cam}_self_mask"
    )
    assert (
        CameraParameterSensor(camera_name=cam, img_resolution=(W, H)).uuid == f"sensor_param_{cam}"
    )

    # ... and the bundle agrees with all of them, by construction rather than by
    # two lists that happen to match.
    bundled = get_perception_sensors(
        _ExpCfg((cam,)), depth=True, segmentation=True, self_mask=True
    )
    bare = {
        CameraParameterSensor(camera_name=cam, img_resolution=(W, H)).uuid,
        DepthSensor(camera_name=cam, img_resolution=(W, H)).uuid,
        SegmentationSensor(camera_name=cam, img_resolution=(W, H)).uuid,
        SelfOcclusionMaskSensor(camera_name=cam, img_resolution=(W, H)).uuid,
    }
    assert {s.uuid for s in bundled} == bare


def test_uuid_templates_are_exactly_these():
    """Full-set equality, so a NEW stream cannot be added without updating the
    canonical list the doc tests below check against."""
    sensors = get_perception_sensors(
        _ExpCfg(("head_camera",)), depth=True, segmentation=True, self_mask=True
    )
    expected = {t.format(cam="head_camera") for t in CANONICAL_UUIDS}
    assert {s.uuid for s in sensors} == expected


def test_code_docstrings_advertise_only_real_uuids():
    """The docstrings a client reads before writing a subscribe list."""
    sources = {
        "get_perception_sensors": get_perception_sensors.__doc__,
        "bridge/client.py": (_REPO / "molmo_spaces" / "bridge" / "client.py").read_text(),
        # The module that DEFINES the uuids is guarded too -- it is where a second
        # scheme would be reintroduced.
        "env/sensors_cameras.py": (
            _REPO / "molmo_spaces" / "env" / "sensors_cameras.py"
        ).read_text(),
    }
    for name, text in sources.items():
        bogus = _documented_uuids(text) - CANONICAL_UUIDS
        assert not bogus, (
            f"{name} documents uuid(s) the code does not produce: {sorted(bogus)}. "
            f"Real ones are {sorted(CANONICAL_UUIDS)}. A client copying this gets a "
            f"silently empty stream."
        )


def test_markdown_docs_advertise_only_real_uuids():
    """Same guard for the prose docs. Skipped rather than failed if a file moves:
    the contract is about what the docs SAY, not about which files exist."""
    for rel in ("docs/bridge.md", "GUIDE.md"):
        path = _REPO / rel
        if not path.is_file():
            continue
        bogus = _documented_uuids(path.read_text()) - CANONICAL_UUIDS
        assert not bogus, f"{rel} documents uuid(s) the code does not produce: {sorted(bogus)}"
