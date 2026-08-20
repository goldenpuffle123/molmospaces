"""The perception bundle: posed depth (+segmentation, +self-mask) for mappers.

`get_nav_task_sensors` attaches RGB and camera parameters only, so a mapping
stack driving a nav task has to bypass the observation dict entirely and call
`env.render_depth_frame` itself. `get_perception_sensors` is the composable
bundle that closes that, and these tests pin its contract.
"""

import numpy as np
import pytest

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors import get_perception_sensors
from molmo_spaces.env.sensors_cameras import SegmentationSensor, SelfOcclusionMaskSensor

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
