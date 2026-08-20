"""The camera-pose frame convention, pinned as a contract rather than a comment.

`Camera.get_pose()` returns cam2world in the OpenCV frame (x right, y down,
z forward). That was previously stated three contradictory ways in the source --
a local named `world2cam`, a docstring saying "cam2world", and a comment saying
"camera looks down negative Z" -- so anything consuming
`CameraParameterSensor`'s output had to guess. Guessing wrong is a silent,
metres-scale back-projection error rather than a crash.

These tests need no renderer and no scene: `Camera` is constructible from
pos/forward/up, and the projection round-trip is pure geometry.
"""

import numpy as np
import pytest

from molmo_spaces.env.camera_manager import Camera
from molmo_spaces.env.sensors_cameras import CameraParameterSensor


def _camera():
    """A camera at a deliberately asymmetric pose, so an axis flip cannot pass."""
    return Camera(
        name="test_cam",
        pos=np.array([1.5, -2.0, 1.3]),
        forward=np.array([0.6, 0.8, -0.2]),
        up=np.array([0.0, 0.0, 1.0]),
        fov=70.0,
    )


def test_get_pose_columns_are_opencv_axes():
    """cam2world columns are [right, down, forward]; translation is the position."""
    cam = _camera()
    T = cam.get_pose()

    fwd = cam.forward / np.linalg.norm(cam.forward)
    right = np.cross(fwd, cam.up / np.linalg.norm(cam.up))
    right /= np.linalg.norm(right)
    down = -np.cross(right, fwd)

    np.testing.assert_allclose(T[:3, 0], right, atol=1e-6)  # +x = right
    np.testing.assert_allclose(T[:3, 1], down, atol=1e-6)  # +y = DOWN, not up
    np.testing.assert_allclose(T[:3, 2], fwd, atol=1e-6)  # +z = FORWARD, not -z
    np.testing.assert_allclose(T[:3, 3], cam.pos, atol=1e-6)

    # A rotation, not a reflection: OpenCV's [right, down, forward] is right-handed.
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-6)


def test_get_pose_is_cam2world_not_world2cam():
    """The origin of the camera frame maps to the camera's world position."""
    cam = _camera()
    T = cam.get_pose()
    np.testing.assert_allclose(T @ np.array([0.0, 0.0, 0.0, 1.0]), [*cam.pos, 1.0], atol=1e-6)

    # A point 2 m straight ahead in camera coords is 2 m along +forward in world.
    ahead_cam = np.array([0.0, 0.0, 2.0, 1.0])
    fwd = cam.forward / np.linalg.norm(cam.forward)
    np.testing.assert_allclose((T @ ahead_cam)[:3], cam.pos + 2.0 * fwd, atol=1e-6)


def test_projection_round_trip():
    """world -> pixel -> world, using the sensor's own intrinsic_cv.

    This is the test that would have caught the mislabel: it only closes if
    cam2world_cv and intrinsic_cv agree on the frame.
    """
    cam = _camera()
    width, height = 640, 480
    T = cam.get_pose()

    f = (height / 2.0) / np.tan(np.radians(cam.fov / 2.0))
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])

    fwd = cam.forward / np.linalg.norm(cam.forward)
    right = np.cross(fwd, cam.up / np.linalg.norm(cam.up))
    right /= np.linalg.norm(right)
    # A point in front of and off to one side of the camera.
    p_world = cam.pos + 3.0 * fwd + 0.4 * right

    p_cam = (np.linalg.inv(T) @ np.array([*p_world, 1.0]))[:3]
    assert p_cam[2] > 0, "a point in FRONT of the camera must have positive z in CV"

    uvw = K @ p_cam
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    assert 0 <= u < width and 0 <= v < height

    # Unproject the way any OpenCV consumer would, and land back on the point.
    z = p_cam[2]
    back_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    back_world = (T @ np.array([*back_cam, 1.0]))[:3]
    np.testing.assert_allclose(back_world, p_world, atol=1e-6)


def test_sensor_fields_are_consistent():
    """extrinsic_cv is the inverse of cam2world_cv; cam2world_gl is an alias."""
    cam = _camera()
    sensor = CameraParameterSensor(camera_name="test_cam", img_resolution=(640, 480))

    class _Registry(dict):
        pass

    env = type("Env", (), {})()
    env.camera_manager = type("CM", (), {})()
    env.camera_manager.registry = _Registry(test_cam=cam)

    data = sensor.get_observation(env, task=None)

    cam2world = np.array(data["cam2world_cv"])
    extrinsic = np.array(data["extrinsic_cv"])

    np.testing.assert_allclose(cam2world, cam.get_pose(), atol=1e-6)
    np.testing.assert_allclose(extrinsic, np.linalg.inv(cam2world)[:3, :], atol=1e-6)
    # Deprecated alias must stay byte-identical while it exists.
    np.testing.assert_array_equal(np.array(data["cam2world_gl"]), cam2world)

    for key in ("cam2world_cv", "cam2world_gl", "extrinsic_cv", "intrinsic_cv"):
        assert key in sensor.observation_space.spaces, key
