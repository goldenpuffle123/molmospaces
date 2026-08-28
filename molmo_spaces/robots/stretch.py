"""Hello Robot Stretch: a differential-drive base with a mast-mounted head.

Ported out of an external next-best-view stack, where it was maintained beside
the planner that drives it. It earns a place next to RBY1 precisely because it is
NOT RBY1: a genuinely nonholonomic drivetrain, a much smaller footprint, an
ACTUATED head (RBY1's is hard-disabled), and a portrait-mounted D435i whose wide
87 deg axis is VERTICAL. It exercises code paths a holonomic robot with a fixed
head never reaches.

WHAT stretch.xml LACKS, added at attach time via HoloBaseKit rather than by
editing the model: the two sites the base group reads, the three base position
actuators (the model authors the planar joints but no actuators -- it is driven
kinematically elsewhere), and the head camera (it declares none at all).

BASE MODE, stated: the base is commanded as `holo_joint_planar_position`, the
absolute [x, y, theta] seam. Stretch is NONHOLONOMIC and nothing in that mode
enforces it, so the config declares `base_kinematics="differential"` and the
constraint is the PLANNER's to respect. The residual is that the sim would not
resist a lateral push, since nothing ever commands one; a genuine wheel-velocity
controller is the honest upgrade if contact dynamics ever matter here.

THE MODEL IS VENDORED at `molmo_spaces/robots/models/stretch` (MJCF + meshes +
clearpath's collision spheres/exclusions), so the port resolves with nothing set.
`MLSPACES_STRETCH_DIR` still overrides it, and `robot_dir` on the config overrides
both -- that is the seam nbv uses to read its own clearpath checkout in place.
"""

import os
from pathlib import Path

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

# Resolution order: `robot_dir` on the config > MLSPACES_STRETCH_DIR > vendored.
# The vendored copy is byte-identical to clearpath's `assets/robots/stretch`,
# minus `stretch.usd` (Isaac-only) and `stretch.xacro` (the ROS source this MJCF
# was generated from) -- neither is read by this stack.
VENDORED_STRETCH_DIR = Path(__file__).parent / "models" / "stretch"
_ENV_DIR = os.environ.get("MLSPACES_STRETCH_DIR", "")
STRETCH_DIR = Path(_ENV_DIR) if _ENV_DIR else VENDORED_STRETCH_DIR

BASE_JOINTS = ("x_to_world_joint", "y_to_x_joint", "base_to_y_joint")
BASE_ACTS = tuple(f"{j[:-6]}_act" for j in BASE_JOINTS)  # <name>_joint -> <name>_act
HEAD_JOINTS = ("joint_head_pan", "joint_head_tilt")
HEAD_ACTS = tuple(f"{j}_position" for j in HEAD_JOINTS)  # stretch.xml's own names
ARM_JOINTS = ("joint_lift", "joint_arm", "joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll")
ARM_ACTS = tuple(f"{j}_position" for j in ARM_JOINTS)

# DRIVING STOW (decision 3): lift, extension, wrist yaw/pitch/roll.
#
# THE LIFT IS THE LOAD-BEARING PART, and it is not about tidiness. At lift 0 the
# arm sits INSIDE base_link's collision mesh -- which, as clearpath's own
# stretch.toml warns, "wraps the entire chassis + telescoping mast in one piece"
# (they omit base_link from their capsules and disable self-collision for this
# robot outright). MEASURED here at the MJCF's OWN zero pose: 7 self-contacts,
# base_link <-> link_wrist_yaw penetrating 0.067 m. That is not cosmetic: the
# constraint solver resists it, and with the arm servo holding the pose the whole
# robot JAMS -- the base actuators saturate (300 N, 300 N, 60 Nm) at exactly zero
# qvel, and the searcher commands 2.51 m of travel while achieving 0.01 m. It
# looked like a weak servo; the servo is fine (0.1 m commanded -> 0.092 m in one
# 0.2 s step on the bare model, 24.7 kg).
#
# MEASURED, self-contacts and envelope over the (lift, wrist_yaw) grid:
#
#     lift 0.00: 7-15 self-contacts at EVERY wrist angle (worst -0.080 m)
#     lift >= 0.20: ZERO self-contacts at every wrist angle
#     arm radial extent at lift 0.35: wrist_yaw 0.0 -> 0.436 m (the gripper),
#                 1.0 -> 0.381, 1.5 -> 0.323, >= 2.0 -> 0.320 (link_lift, a
#                 plateau: folding further buys nothing)
#
# So: lift 0.35 (well clear of the 0.20 threshold, mid-range of [0, 1.1]) and
# wrist_yaw 2.0 (on the plateau, inside [-1.75, 4.0]). Extension stays retracted;
# the gripper tendon is not in this group and rests at ctrl 0 = closed.
#
# LESSON, recorded because it cost a debugging cycle: an envelope minimised on
# RADIUS ALONE picked wrist_yaw 3.0 at lift 0, which is the most compact pose AND
# the most deeply self-colliding one (15 contacts, -0.080 m). Compactness and
# feasibility are different questions; measure both.
STOW_Q = (0.35, 0.0, 2.0, 0.0, 0.0)

# Base servo gains + travel, synthesized because the MJCF has no base actuators.
# The x/y slides are UNBOUNDED in the model; stretch.toml's [constraints] lbx/ubx
# give the virtual base +-100 m, which is what BASE_XY_RANGE mirrors. Yaw takes
# the joint's own +-pi. ctrlrange is not decoration -- see the module docstring.
BASE_KP, BASE_KV = 5000.0, 500.0
BASE_XY_RANGE = (-100.0, 100.0)
BASE_YAW_RANGE = (-3.14159, 3.14159)

# Head camera = the D435i DEPTH-OPTICAL frame in link_head_tilt, taken VERBATIM
# from clearpath's nbv_bridge (composed there from the stretch.urdf fixed-joint
# chain camera_joint -> camera_link_joint -> camera_depth_joint ->
# camera_depth_optical_joint, converted to MuJoCo's camera convention: looks
# along -z, +y up). The +92 deg image roll is INSIDE this quaternion -- it is the
# real portrait mount, not a transcription slip -- which is why HEAD_FOVY is the
# sensor's WIDE 87 deg axis and the render is portrait (decision 2).
D435_POS = [0.035952, 0.006302, 0.032056]
D435_QUAT = [-0.489796, -0.503081, 0.50652, 0.500447]  # wxyz
HEAD_FOVY = 87.0
PORTRAIT_WH = (480, 640)  # MolmoSpaces img_resolution is (width, height)
PORTRAIT_HW = (640, 480)  # nbv render_hw is (rows, cols)

# Head pose. Pan 0 = straight ahead. The TILT is frozen from a MEASUREMENT, not
# guessed: a scouting probe swept joint_head_tilt over its whole range on the
# BUILT scene and fitted the (near-linear) tilt -> world gaze-pitch map. (The
# probe is deleted per the repo's probe convention; its findings are these.)
#
#     MEASURED 2026-08-17:  pitch_deg = 57.259 * tilt + 2.532
#     reachable gaze:       -82.2 deg (tilt -1.53) .. +47.8 deg (tilt +0.79)
#     sign:                 NEGATIVE tilt = looking DOWN  (the Hello Robot
#                           convention this port assumed, now confirmed)
#
# So a different HEAD_PITCH_DEG needs no probe re-run, just the inverse:
# tilt = (pitch_deg - 2.532) / 57.259. The policy then re-reads the pitch it
# ACTUALLY got and locks the planner to that, so a stale constant here shows up
# as a different (still self-consistent) gaze, never as a plan/execute mismatch.
HEAD_PITCH_DEG = -25.0
HEAD_TILT = -0.4808  # rad, = the fit above inverted at -25 deg
HEAD_PAN = 0.0
# The fit above, as the two numbers the unlock needs: tilt = (pitch - B) / A, and
# the joint's own travel. GAZE UNLOCK (2026-08-17, superseding decision 1's
# "locked for now"): the planner already carries gaze pitch as a control
# dimension in every backend -- it is the tail scalar, step[-1] -- so unlocking
# is not a new mechanism, it is letting that dimension reach the actuator and
# dropping the two knobs that pinned it (sigma_pitch 0, pitches_deg = the one
# measured value). nbv's Stretch backend already clips the CEM to this robot's
# real head travel via gaze_pitch = (-80, 10), which was written for exactly this
# day. Pan stays at 0: nbv has no pan dimension, so adding one is a dynamics
# change, not an unlock -- the base turns instead, as it always has.
PITCH_A, PITCH_B = 57.259, 2.532
TILT_RANGE = (-1.53, 0.79)  # rad, stretch.toml [constraints] head-tilt lbx/ubx

# GROUND LIFT [m]. MEASURED: stretch.xml is authored
# flush to the floor -- the wheel and base_link collision meshes reach z = -0.0000
# exactly -- so at every candidate pose MuJoCo reports robot-vs-ground contacts.
# MolmoSpaces' placement test forgives ground contact only when the other body's
# ROOT NAME contains "floor" (env.check_robot_collision_in_current_pose), and in
# a ProcTHOR house the floor belongs to the merged `world` body, so the forgiveness
# never fires and EVERY placement is rejected (measured: 0/10 attempts, then
# HouseInvalidForTask; the contacts were link_right_wheel/link_left_wheel/base_link
# vs `world`). The omni base does not hit this because its own chassis geometry
# floats clear of the plane.
#
# Lifting the kinematic chain removes that contact channel and nothing else: the
# base has NO vertical DOF (x slide -> y slide -> yaw hinge), so the robot cannot
# fall, and gravcomp is on. 0.02 m rather than the 0.005 first tried: at 5 mm the
# wheels still touched, MEASURED in a live episode as
# link_left_wheel <-> world dist = -0.0023, i.e. this house's floor surface sits
# ~7 mm above z = 0. Contact there is pure drag on a kinematically driven base.
# The cost, stated: a floor lip or doorsill below 2 cm is now passed through
# rather than hit.
GROUND_LIFT = 0.02

# Planar-drive authority. Same lesson ms_arm records: a base servo strong enough
# to hold position under mj_forward will BURY itself in a wall instead of being
# stopped by one, and once buried the whole robot freezes.
#
# The first values here (300 N / 60 Nm, scaled off the omni base's 400/100) LOST
# that bet and the failure was measured, not argued: on procthor-10k-train house 4
# the base drove into an unmapped wall and burrowed 0.085 m in, then sat there for
# 420 frames -- commanded 7.4 m per 20-frame window, achieved 0.00 m, planner
# perfectly healthy the whole time (score ~0.37, hold 0), the bumper reversing 22
# times without recovering a millimetre.
#
# So size them to the PLATFORM instead of to the other robot. Stretch is 24.74 kg
# with base yaw inertia ~2.7 kg m^2, and the servo only has to reach the platform
# limits within one control period:
#
#     translation  m*v_max/dt = 24.74 * 0.5 / 0.2 =  62 N   (need)
#                  mu*m*g     = 0.7 * 24.74 * 9.81 = 170 N  (wheel traction, cap)
#     yaw          I*omega_max/dt = 2.7 * 1.5 / 0.2 = 20 Nm (need)
#
# 100 N gives 4.0 m/s^2 -- comfortably above the 2.5 m/s^2 that 62 N buys, and
# well under the traction bound, so the wall wins the argument instead of the
# actuator. 300 N was 12.1 m/s^2 on a robot whose top speed is 0.5 m/s: ~5x more
# authority than the drivetrain physically has.
#
# MEASURED CAVEAT, worth knowing before tuning these further: the base saturates
# its force limit even in CONTACT-FREE driving (peak |force| = the cap on every
# axis), because the lookahead carrot commands a target ~0.2-1.0 m ahead into a
# kp=5000 servo. The drive is effectively bang-bang, so this cap sets the
# acceleration, not just the collision behaviour.
BASE_FORCE, BASE_TORQUE = 100.0, 20.0


# --------------------------------------------------------------------------- #
# Move groups + robot view
# --------------------------------------------------------------------------- #
class StretchBaseGroup(HoloJointsRobotBaseGroup):
    """The planar base. The MJCF ships the three virtual joints but NO actuators
    for them (clearpath drives that base kinematically), so both the sites and the
    actuators are synthesized in add_robot_to_scene.

    HINGE STOP, inherited from the RBY1/omni lesson: base_to_y_joint is a hinge
    driven by a +-pi position actuator, and the ctrl setter wraps the target to
    within pi of the CURRENT yaw -- so a shortest-turn command near +-pi lands
    past the stop and the servo stops tracking. ms_nav's `_yaw_to` guard prevents
    that and is inherited unchanged."""

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
        # Compiled-model check, so a model that ships its own planar actuators
        # without ranges fails here too -- not only one built by the kit.
        HoloBaseKit.assert_planar_ctrlrange(m, [f"{namespace}{a}" for a in BASE_ACTS])


class StretchHeadGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """Head pan + tilt, on the MJCF's OWN position actuators.

    It exists even though decision 1 never commands it, for three concrete
    reasons: init_qpos is keyed by move group (so this is how the head reaches
    HEAD_TILT at all), an uncommanded group is HELD by Robot.update_control ->
    set_to_stationary (so the tilt stays where reset put it, rather than being
    dragged to the actuator's ctrl 0 = level gaze), and unlocking the gaze later
    is then a change to the action dict rather than a new move group."""

    def __init__(self, mj_data, base: RobotBaseGroup, namespace: str = "") -> None:
        m = mj_data.model
        self._tilt_id = m.body(f"{namespace}link_head_tilt").id
        self._root_id = m.body(f"{namespace}link_head_pan").id
        super().__init__(
            mj_data,
            [m.joint(f"{namespace}{j}").id for j in HEAD_JOINTS],
            [m.actuator(f"{namespace}{a}").id for a in HEAD_ACTS],
            self._root_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._tilt_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)


class StretchArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """Lift + telescoping extension + the 3 wrist joints, on the MJCF's own
    position actuators. It exists ONLY to hold STOW_Q (decision 3): never
    commanded, so update_control freezes it where reset put it. No gripper -- the
    2F-style tendon actuator rests at ctrl 0 (closed), which is what driving
    wants anyway."""

    def __init__(self, mj_data, base: RobotBaseGroup, namespace: str = "") -> None:
        m = mj_data.model
        self._leaf_id = m.body(f"{namespace}link_wrist_roll").id
        self._root_id = m.body(f"{namespace}link_lift").id
        super().__init__(
            mj_data,
            [m.joint(f"{namespace}{j}").id for j in ARM_JOINTS],
            [m.actuator(f"{namespace}{a}").id for a in ARM_ACTS],
            self._root_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)


class StretchRobotView(RobotView):
    """base + head + a stow-holding arm (decision 3). No gripper group: nothing
    here grasps, and its tendon rests closed."""

    def __init__(self, mj_data, namespace: str = "") -> None:
        self._namespace = namespace
        base = StretchBaseGroup(mj_data, namespace=namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "head": StretchHeadGroup(mj_data, base, namespace=namespace),
                "arm": StretchArmGroup(mj_data, base, namespace=namespace),
            },
        )

    @property
    def name(self) -> str:
        return f"{self._namespace}stretch"

    @property
    def base(self) -> StretchBaseGroup:
        return self._move_groups["base"]


# --------------------------------------------------------------------------- #
# The robot
# --------------------------------------------------------------------------- #
class StretchRobot(Robot):
    """MolmoSpaces Robot for the Stretch: a position-commanded planar base and a
    position-commanded head. The ACTION is {"base": [x, y, theta]}; omitting
    "head" holds the head where reset put it (Robot.update_control ->
    set_to_stationary), which is exactly decision 1."""

    def __init__(self, mj_data, exp_config) -> None:
        super().__init__(mj_data, exp_config)
        rc = exp_config.robot_config
        self._namespace = rc.robot_namespace
        self._robot_view = StretchRobotView(mj_data, self._namespace)
        assert rc.command_mode.get("base") == "holo_joint_planar_position", rc.command_mode
        assert rc.command_mode.get("head") == "joint_position", rc.command_mode
        assert rc.command_mode.get("arm") == "joint_position", rc.command_mode
        self._controllers = {
            k: JointPosController(self._robot_view.get_move_group(k))
            for k in ("base", "head", "arm")
        }

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        # The nav task registers no EE-pose/twist sensor, so nothing here needs
        # IK. Raise rather than silently building a second model.
        raise NotImplementedError("stretch: no IK solver wired (nav needs none)")

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("stretch: no parallel kinematics")

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def get_arm_move_group_ids(self) -> list[str]:
        # EMPTY on purpose though an "arm" move group exists: this list is what
        # MolmoSpaces treats as manipulators (action noise, IK, grasp paths), and
        # decision 3 says this arm is furniture -- it holds STOW_Q and nothing
        # else. Listing it would invite machinery that has no meaning here.
        return []

    def apply_action_noise(self, action):
        # Uncalibrated noise would quietly become part of every number. OFF,
        # stated -- same call as ms_arm.
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
        """Attach, then add the three things stretch.xml does not have: the base
        group's two sites, the three base position actuators, and the head camera
        at the D435i frame.

        Insertion is pinned to the origin, unrotated, like RBY1 and the omni base:
        the base joints are WORLD-AXIS slides, so a rotated attach frame would
        make a commanded (x, y) mean something else."""
        assert np.allclose(np.asarray(pos, float)[:2], 0.0), (
            f"stretch must be inserted at the origin (got {pos})"
        )
        assert np.allclose(np.asarray(quat, float), [1, 0, 0, 0]), (
            f"stretch must be inserted unrotated (got {quat})"
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
        root = spec.body(f"{prefix}{cls.robot_model_root_name()}")
        root.pos = np.array([root.pos[0], root.pos[1], root.pos[2] + GROUND_LIFT], float)
        HoloBaseKit.add_base_sites(spec, prefix, "base_link")
        HoloBaseKit.add_planar_actuators(
            spec,
            prefix,
            BASE_JOINTS,
            BASE_ACTS,
            xy_range=BASE_XY_RANGE,
            theta_range=BASE_YAW_RANGE,
            kp=BASE_KP,
            kv=BASE_KV,
            force=robot_config.base_force,
            torque=robot_config.base_torque,
        )
        HoloBaseKit.add_head_camera(
            spec, prefix, "link_head_tilt", "head_cam", D435_POS, D435_QUAT, HEAD_FOVY
        )


class StretchConfig(BaseRobotConfig):
    """Robot config. `robot_dir` is BaseRobotConfig's seam for a robot that is not
    a prepackaged MlSpaces asset, so nothing is installed into their tree."""

    robot_cls: type = StretchRobot
    robot_factory: object = StretchRobot
    robot_view_factory: object = StretchRobotView
    robot_namespace: str = "robot_0/"
    name: str = "stretch"
    robot_dir: Path | None = STRETCH_DIR
    robot_xml_path: Path = Path("stretch.xml")
    init_qpos: dict = {"base": [0.0, 0.0, 0.0], "head": [HEAD_PAN, HEAD_TILT], "arm": list(STOW_Q)}
    # No head noise: the gaze is LOCKED and the planner is told the measured
    # pitch, so per-episode tilt noise would only add a nuisance variable.
    init_qpos_noise_range: dict | None = None
    command_mode: dict = {
        "base": "holo_joint_planar_position",
        "head": "joint_position",
        "arm": "joint_position",
    }
    gravcomp: bool = True
    base_force: float = BASE_FORCE  # [N] per translation axis
    base_torque: float = BASE_TORQUE  # [Nm] yaw
    # DECLARED, not enforced -- see BaseRobotConfig.base_kinematics. This robot
    # cannot strafe; the command mode would let it.
    base_kinematics: str = "differential"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        assert self.robot_dir is not None, (
            "stretch model directory unknown: robot_dir was explicitly set to "
            "None, which defeats both the vendored model and MLSPACES_STRETCH_DIR."
        )
        assert self.get_robot_xml_path().is_file(), (
            f"stretch model not found at {self.get_robot_xml_path()}"
        )


class StretchCameraSystem(CameraSystemConfig):
    """The head D435i, PORTRAIT (decision 2). `head_camera` is the registry name
    the nav task's success test reads, so it stays that whatever it is mounted on.

    fov is pinned to the MJCF's own 87 rather than left None, so a model change
    shows up as an assertion at reset instead of as a silently different lens."""

    img_resolution: tuple = PORTRAIT_WH
    cameras: list = [
        MjcfCameraConfig(
            name="head_camera",
            mjcf_name="head_cam",
            robot_namespace="robot_0/",
            fov=HEAD_FOVY,
            record_depth=True,
        ),
    ]
