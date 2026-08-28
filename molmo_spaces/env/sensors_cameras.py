"""Camera-derived sensors: RGB, depth, segmentation, self-occlusion, parameters.

UUID CONVENTION, one scheme, stated once. An observation dict is keyed by sensor
uuid, and consumers -- ``BridgePolicy``'s subscription filter, the HDF5 writer's
``sensor_param_`` grouping, every external client -- match those keys exactly. A
name nothing produces is not rejected anywhere; it simply yields nothing, which
looks like a broken sensor.

So the DEFAULT uuid of every sensor here is the name production actually uses::

    {cam}                    RGB
    {cam}_depth              metric z-depth [m]
    {cam}_segmentation       [H, W, 3], channel 2 is the body id
    {cam}_self_mask          the robot's own body
    sensor_param_{cam}       cam2world_cv + intrinsic_cv

These previously defaulted to a second scheme built from the ``depth_`` and
``camera_params_`` prefixes, which no caller ever used, because every bundle in
``env/sensors.py`` and ``env/rby1_sensors.py`` passes ``uuid=`` explicitly. Two
schemes with only one of them real is how the docs came to advertise uuids that
never appear in an observation; the defaults now agree with the bundles, so a
bare construction and a bundled one produce the same key.
"""

import gymnasium.spaces as gyms
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor


class CameraSensor(Sensor):
    """Sensor for RGB camera images from MuJoCo."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = camera_name

        # Define observation space for RGB images
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get camera image from environment rendering."""

        # Use camera-specific frame access for multi-camera support
        # if hasattr(env, 'render_rgb_frame') and callable(env.render_rgb_frame):
        frame = env.render_rgb_frame(self.camera_name)

        if frame is not None:
            return frame

        # Return black image if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 3), dtype=np.uint8)


class DepthSensor(Sensor):
    """Sensor for depth images from MuJoCo.

    Returns raw metric depth in meters as float32. Encoding to RGB for video storage
    happens at save time. See molmo_spaces.utils.depth_utils for encoding/decoding functions.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"{camera_name}_depth"

        # Define observation space for raw depth (float32 in meters)
        width, height = img_resolution
        observation_space = gyms.Box(low=0.0, high=10.0, shape=(height, width), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get depth image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "render_depth_frame") and callable(env.render_depth_frame):
            frame = env.render_depth_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "depth_frame") and env.depth_frame is not None:
            return env.depth_frame

        # Return zero depth if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width), dtype=np.float32)


class SegmentationSensor(Sensor):
    """Sensor for segmentation images from MuJoCo, outputs video-compatible arrays."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"{camera_name}_segmentation"

        # Define observation space for uint8 images with channel dimension
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get segmentation image from environment rendering.

        Renders per camera via ``render_segmentation_frame``, mirroring
        DepthSensor. The previous implementation called ``env.segmentation_frame``
        as if it were a method, but it is a read-only property backed by
        ``_segmentation_frame``, which is initialised to None and never assigned
        -- so both branches failed and this sensor always returned zeros.
        """
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "render_segmentation_frame") and callable(env.render_segmentation_frame):
            frame = env.render_segmentation_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to the last rendered frame, if the env exposes one
        if getattr(env, "segmentation_frame", None) is not None:
            return env.segmentation_frame

        # Return zero segmentation if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 1), dtype=np.uint8)


class SelfOcclusionMaskSensor(Sensor):
    """bool[H, W]: pixels showing the robot's OWN bodies, in one camera.

    Why a mapping stack needs this. A wide, downward-pitched head camera sees a
    large slice of its own robot (RBY1's 139 deg lens pitched ~33 deg down sees
    torso and arms). Those depth returns are real, but they say nothing about the
    world -- integrate them into an occupancy map and the robot writes itself in
    as an obstacle at its own standing height, then refuses to move.

    Pixels flagged here must be DROPPED by the consumer: not marked free, not
    marked occupied. This sensor only reports them; it does not alter depth.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        robot_namespace: str | None = None,
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution
        self.robot_namespace = robot_namespace
        if uuid is None:
            uuid = f"{camera_name}_self_mask"
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=1, shape=(height, width), dtype=bool)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def _robot_body_ids(self, env) -> np.ndarray:
        """Body ids under the robot's namespace, recomputed per scene."""
        import mujoco

        model = env.current_model
        ns = self.robot_namespace
        if ns is None:
            robot = getattr(env, "current_robot", None)
            ns = getattr(robot, "robot_namespace", None) or "robot_0/"
        return np.array(
            [
                b
                for b in range(model.nbody)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(ns)
            ],
            dtype=np.int32,
        )

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        seg = env.render_segmentation_frame(self.camera_name)
        if seg is None:
            width, height = self.img_resolution
            return np.zeros((height, width), dtype=bool)
        seg = np.asarray(seg)
        # Channel 2 of MolmoSpaces' segmentation frame carries the body id.
        body_ids = seg[..., 2] if seg.ndim == 3 and seg.shape[-1] >= 3 else seg
        return np.isin(body_ids, self._robot_body_ids(env))


class CameraParameterSensor(Sensor):
    """Sensor for camera parameters (intrinsics and extrinsics).

    Frame convention, stated once because it is easy to get wrong:

    * ``cam2world_cv`` maps CAMERA points to WORLD points, in the OpenCV frame
      (x right, y down, z forward) -- the frame ``intrinsic_cv`` assumes.
    * ``extrinsic_cv`` is its inverse (world -> camera), top 3 rows.
    * ``cam2world_gl`` is a DEPRECATED alias of ``cam2world_cv``, kept because
      existing recordings and readers use the key. It was never GL-framed;
      consumers wanting OpenGL must post-multiply by ``diag(1, -1, -1, 1)``.
    """

    def __init__(
        self,
        camera_name: str,
        img_resolution: tuple[int, int],
        uuid: str | None = None,
    ) -> None:
        self.img_resolution = img_resolution
        self.camera_name = camera_name

        if uuid is None:
            uuid = f"sensor_param_{camera_name}"

        observation_space = gyms.Dict(
            {
                "extrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32),
                "cam2world_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32),
                # deprecated alias of cam2world_cv (see class docstring)
                "cam2world_gl": gyms.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32),
                "intrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32),
            }
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        """Get camera parameters for a specific environment."""
        camera = env.camera_manager.registry[self.camera_name]
        # get_pose() returns cam2world in the OpenCV frame (see Camera.get_pose).
        cam2world_cv = camera.get_pose()
        # extrinsic_cv is the inverse: world -> camera, top 3 rows.
        extrinsic_cv = np.linalg.inv(cam2world_cv)[:3, :]  # 3x4 matrix

        width, height = self.img_resolution
        fovy_degrees = camera.fov

        # Convert field of view to focal length
        focal_length = (height / 2.0) / np.tan(np.radians(fovy_degrees / 2.0))

        # Create intrinsic matrix (assuming square pixels and centered principal point)
        intrinsic_cv = np.array(
            [[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )

        # Ensure consistent structure and ordering
        cam2world_list = cam2world_cv.tolist()
        data = {
            "cam2world_cv": cam2world_list,
            # DEPRECATED alias, identical payload. Kept so existing readers and
            # recordings keep working; prefer cam2world_cv in new code.
            "cam2world_gl": cam2world_list,
            "extrinsic_cv": extrinsic_cv.tolist(),
            "intrinsic_cv": intrinsic_cv.tolist(),
        }
        return data
