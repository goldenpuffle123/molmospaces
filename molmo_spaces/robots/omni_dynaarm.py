"""Clearpath omni base + a 6-DOF DynaArm with an eye-in-hand camera.

The mobile-manipulation-for-search embodiment, ported out of an external
next-best-view stack: a holonomic base that can strafe, an arm whose camera can
be driven into cavities a fixed head never sees, and a second fixed deck camera
so the nav task's success test always has a `head_camera` to read.

Like the Stretch port, the MJCF is missing the base sites and the three planar
actuators, and they are added at attach time through HoloBaseKit rather than by
editing the model. The arm's actuators are synthesized for the same reason (the
model deliberately declares none).

THE MODEL IS NOT VENDORED. Point `MLSPACES_OMNI_DYNAARM_DIR` at a directory
holding `omni_base_dynaarm.xml`, or set `robot_dir` on the config.

STATUS: ported and import-checked, NOT exercised in an episode here.
"""

import os
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.holo_base_kit import HoloBaseKit
from molmo_spaces.robots.robot_views.abstract import (
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

_ENV_DIR = os.environ.get("MLSPACES_OMNI_DYNAARM_DIR", "")
OMNI_DYNAARM_DIR = Path(_ENV_DIR) if _ENV_DIR else None

BASE_JOINTS = ("x_to_world_joint", "y_to_x_joint", "base_to_y_joint")
BASE_ACTS = ("x_to_world_act", "y_to_x_act", "base_to_y_act")
# The DynaArm chain, base -> tip. INLINED rather than imported from a planner's
# vendored copy: this module must not depend on the stack that drives it, and the
# ORDER is a contract -- a planner's joint vector is this order.
ARM_JOINTS = (
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "forearm_rotation",
    "wrist_flexion",
    "wrist_rotation",
)
# Driving stow. Plain floats: the config is serialized per episode.
STOW = [0.0, -1.0, 3.0, 0.0, 1.0, 0.0]
# Joint limits, same order as ARM_JOINTS.
ARM_JOINT_RANGES = (
    (-4.31239, 4.31239),
    (-1.7208, 1.7208),
    (0.0, 3.09159),
    (-4.71239, 4.71239),
    (-1.8208, 1.8208),
    (-4.71239, 4.71239),
)
# Synthesized arm servo gains, clearpath's `configure_position_actuators` values
# (the MJCF deliberately declares no arm actuators, see its header comment).
ARM_KP, ARM_KV = 5000.0, 500.0


# --------------------------------------------------------------------------- #
# Move groups + robot view
# --------------------------------------------------------------------------- #
class OmniArmBaseGroup(HoloJointsRobotBaseGroup):
    """The holonomic base. Unlike RBY1/MobileFranka -- whose x/y/theta actuators
    MolmoSpaces synthesizes as SITE actuators -- this robot ships its own planar
    joints AND their position actuators in the MJCF, so only the two sites the
    group's ctor needs are added at attach time (add_robot_to_scene).

    HINGE STOP, inherited from the RBY1 lesson: base_to_y_joint is
    range="-3.14159 3.14159" driven by a ctrlrange +-3.1416 position actuator,
    and the ctrl setter wraps the target to within pi of the CURRENT yaw -- so a
    shortest-turn command near +-pi lands past the joint stop and the servo stops
    tracking. ms_nav's `_yaw_to` guard (joint-frame arithmetic, clipped to
    +-YAW_SAFE) is what prevents that, and it is inherited unchanged."""

    def __init__(self, mj_data, namespace: str = "") -> None:
        m = mj_data.model
        super().__init__(
            mj_data,
            m.site(f"{namespace}world").id,
            m.site(f"{namespace}base_site").id,
            [m.joint(f"{namespace}{j}").id for j in BASE_JOINTS],
            [m.actuator(f"{namespace}{a}").id for a in BASE_ACTS],
            m.body(f"{namespace}base_link").id,
        )


class OmniArmArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """The 6-DOF DynaArm. Joint ORDER is the vendored chain's (shoulder_rotation
    ... wrist_rotation), which is what nbv's `q` means -- a permutation here would
    silently mean the planner commands a different arm than it imagines."""

    def __init__(self, mj_data, base: RobotBaseGroup, namespace: str = "") -> None:
        m = mj_data.model
        self._flange_id = m.body(f"{namespace}flange").id
        self._root_id = m.body(f"{namespace}shoulder").id
        super().__init__(
            mj_data,
            [m.joint(f"{namespace}{j}").id for j in ARM_JOINTS],
            [m.actuator(f"{namespace}{j}_act").id for j in ARM_JOINTS],
            self._root_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._flange_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)


class OmniDynaArmRobotView(RobotView):
    """base + arm. NO gripper move group: this is a navigation task, the 2F-85 is
    never commanded, and its tendon actuator simply rests at ctrl 0 (closed). A
    gripper group would be four abstract methods of dead weight -- add it the day
    something grasps."""

    def __init__(self, mj_data, namespace: str = "") -> None:
        self._namespace = namespace
        base = OmniArmBaseGroup(mj_data, namespace=namespace)
        super().__init__(
            mj_data, {"base": base, "arm": OmniArmArmGroup(mj_data, base, namespace=namespace)}
        )

    @property
    def name(self) -> str:
        return f"{self._namespace}omni_base_dynaarm"

    @property
    def base(self) -> OmniArmBaseGroup:
        return self._move_groups["base"]


# --------------------------------------------------------------------------- #
# The robot
# --------------------------------------------------------------------------- #
class OmniDynaArmRobot(Robot):
    """MolmoSpaces Robot for omni_base_dynaarm: a holonomic planar base and a
    6-DOF arm, both position-commanded, so the ACTION is
    {"base": [x, y, theta], "arm": [q0..q5]} -- exactly nbv's dynaarm control
    output, with nothing projected away. Omitting a key holds that group
    (Robot.update_control -> set_to_stationary -> current joint positions), which
    is what the bumper's reverse and the approach's turn-in-place want."""

    def __init__(self, mj_data, exp_config) -> None:
        super().__init__(mj_data, exp_config)
        rc = exp_config.robot_config
        self._namespace = rc.robot_namespace
        self._robot_view = OmniDynaArmRobotView(mj_data, self._namespace)
        assert rc.command_mode.get("base") == "holo_joint_planar_position", rc.command_mode
        assert rc.command_mode.get("arm") == "joint_position", rc.command_mode
        self._controllers = {
            "base": JointPosController(self._robot_view.get_move_group("base")),
            "arm": JointPosController(self._robot_view.get_move_group("arm")),
        }

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        # The nav task registers no EE-pose/twist sensor (get_nav_task_sensors),
        # so nothing in this path needs IK. Raise instead of silently building a
        # second model.
        raise NotImplementedError("omni_base_dynaarm: no IK solver wired (the nav task needs none)")

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("omni_base_dynaarm: no parallel kinematics")

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def get_arm_move_group_ids(self) -> list[str]:
        return ["arm"]

    def apply_action_noise(self, action):
        # Action noise needs a Jacobian per arm move group and a noise model for
        # the base; neither is calibrated for this robot, and injecting an
        # uncalibrated one would quietly become part of every number. OFF, stated.
        return action

    def reset(self) -> None:
        for mg_id, qpos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = np.asarray(qpos)
        for c in self._controllers.values():
            c.reset()

    def get_world_pose_tf_mat(self):
        return self._robot_view.get_move_group("base").pose

    @staticmethod
    def robot_model_root_name() -> str:
        # The planar chain's root: x slide -> y slide -> base_link (yaw hinge).
        # Robot.apply_control_overrides prefixes this with the namespace, which is
        # correct here (unlike RBY1, whose MJCF names already carry robot_0/).
        return "x_prismatic_link"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config,
        spec,
        prefix,
        pos,
        quat,
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        """Attach the robot, then add what MolmoSpaces needs and the MJCF lacks:
        the two sites the holo-base group reads, and the six arm position
        actuators (the model's header says to synthesize them at load).

        Insertion is pinned to the origin with identity rotation, like RBY1: the
        base joints are world-axis slides, so a rotated attach frame would make
        the commanded (x, y) mean something else."""
        assert np.allclose(np.asarray(pos, float)[:2], 0.0), (
            f"omni_base_dynaarm must be inserted at the origin (got {pos})"
        )
        assert np.allclose(np.asarray(quat, float), [1, 0, 0, 0]), (
            f"omni_base_dynaarm must be inserted unrotated (got {quat})"
        )
        super().add_robot_to_scene(
            robot_config=robot_config,
            spec=spec,
            prefix=prefix,
            pos=list(pos),
            quat=list(quat),
            randomize_textures=randomize_textures,
            strip_meshes=strip_meshes,
        )
        HoloBaseKit.add_base_sites(spec, prefix, "base_link")
        # NOTE this MJCF DOES author its own planar actuators, so the kit's
        # add_planar_actuators is not needed here -- only their force limit is
        # overridden below. The ctrlrange check still runs, in the base group.
        # ctrlrange is LOAD-BEARING, not decoration: JointPosController clips its
        # target to the actuator's ctrlrange, which defaults to [0, 0] -- an
        # unlimited actuator would silently drive every joint to zero.
        # BASE FORCE LIMIT -- the measured cure for "the robot gets stuck".
        # The MJCF gives the planar drive +-2040 N on a 34 kg base (~6x its own
        # weight) because its own comment treats the base as "otherwise
        # kinematic": those gains exist to hold position under mj_forward, not to
        # model a drivetrain. In a furnished house such a base does not stop at a
        # wall, it BURIES itself in one -- MEASURED 0.308 m of interpenetration,
        # at which point the constraint solver resists in every direction and
        # nothing moves at all: base x, y, yaw AND all six arm joints frozen under
        # full servo authority, through six 0.8 m extractions. Every policy-level
        # recovery is powerless there, which is why three of them in a row failed.
        # A realistic cap makes contact STOP the robot instead of swallowing it,
        # and leaves the bumper something it can actually pull against. Normal
        # driving is untouched: the planar joints carry no wheel friction, so
        # 400 N still accelerates the ~50 kg body at ~8 m/s^2.
        lim = {
            f"{prefix}x_to_world_act": robot_config.base_force,
            f"{prefix}y_to_x_act": robot_config.base_force,
            f"{prefix}base_to_y_act": robot_config.base_torque,
        }
        for a in spec.actuators:
            if a.name in lim:
                a.forcelimited = True
                a.forcerange = np.array([-lim[a.name], lim[a.name]], float)
        # The chain's joint limits, inlined for the same reason the names are.
        rng = dict(zip(ARM_JOINTS, ARM_JOINT_RANGES, strict=True))
        for j in ARM_JOINTS:
            a = spec.add_actuator()
            a.name = f"{prefix}{j}_act"
            a.target = f"{prefix}{j}"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_position(ARM_KP, ARM_KV)
            a.ctrllimited = True
            a.ctrlrange = np.asarray(rng[j], float)


class OmniDynaArmConfig(BaseRobotConfig):
    """Robot config. `robot_dir` is the BaseRobotConfig seam for a robot that is
    not a prepackaged MlSpaces asset, so nothing is installed into their tree."""

    robot_cls: type = OmniDynaArmRobot
    robot_factory: object = OmniDynaArmRobot
    robot_view_factory: object = OmniDynaArmRobotView
    robot_namespace: str = "robot_0/"
    name: str = "omni_base_dynaarm"
    robot_dir: Path | None = OMNI_DYNAARM_DIR
    robot_xml_path: Path = Path("omni_base_dynaarm.xml")
    init_qpos: dict = {"base": [0.0, 0.0, 0.0], "arm": STOW}
    init_qpos_noise_range: dict | None = None  # the stow pose DEFINES the baked
    #                                             sphere column; noise on it would
    #                                             desynchronize body and model
    command_mode: dict = {"base": "holo_joint_planar_position", "arm": "joint_position"}
    gravcomp: bool = True
    # Planar-drive authority (see add_robot_to_scene). A DEVIATION from the MJCF's
    # +-2040 N / +-204 Nm, stated as such: that model declares the base kinematic,
    # and a kinematic base bulldozes into house geometry instead of being stopped
    # by it. These are the values a 34 kg omni base could plausibly produce.
    base_force: float = 400.0  # [N] per translation axis
    base_torque: float = 100.0  # [Nm] yaw

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        assert self.robot_dir is not None, (
            "omni_base_dynaarm model directory unknown: set "
            "MLSPACES_OMNI_DYNAARM_DIR to a directory containing "
            "omni_base_dynaarm.xml, or pass robot_dir."
        )
        assert self.get_robot_xml_path().is_file(), (
            f"omni_base_dynaarm model not found at {self.get_robot_xml_path()}"
        )


class OmniDynaArmCameraSystem(CameraSystemConfig):
    """Both MJCF cameras. `head_camera` is the REGISTRY name the nav task's
    success test reads (nav_task.py), and here it is the EE camera -- see the
    module docstring, decision 1. `nav_camera` is the fixed deck eye the planner's
    base camera models and the belief slice sits at.

    fov is pinned to the MJCF's own 42 deg rather than left None so a model change
    shows up as an assertion at reset instead of as a silently different lens."""

    img_resolution: tuple = (640, 480)
    cameras: list = [
        MjcfCameraConfig(
            name="head_camera",
            mjcf_name="ee_cam",
            robot_namespace="robot_0/",
            fov=42.0,
            record_depth=True,
        ),
        MjcfCameraConfig(
            name="nav_camera",
            mjcf_name="base_front_cam",
            robot_namespace="robot_0/",
            fov=42.0,
            record_depth=True,
        ),
    ]
