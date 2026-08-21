"""Attach-time fixups every mobile MJCF needs for ``HoloJointsRobotBaseGroup``.

Three things were missing from every mobile model integrated so far, and each
fails in a way that does not look like a missing model feature:

* **the two sites** the base group reads (``world``, ``base_site``). Absent, the
  group cannot be constructed at all -- the loud, easy failure.
* **the planar position actuators**. Models that are driven kinematically
  elsewhere (Stretch is) author the x/y/theta joints and no actuators for them.
  ``ctrlrange`` on those actuators is LOAD-BEARING: ``JointPosController`` clips
  the command to it and MuJoCo defaults it to ``[0, 0]``, so a model that merely
  forgets the range drives the base to the WORLD ORIGIN on the first step and
  holds it there. That reads as a planner bug for as long as you let it.
* **a head camera**. Many mobile MJCFs declare no cameras whatsoever.

These are static helpers rather than a mixin on purpose: they are called from
``add_robot_to_scene``, which is a classmethod on the robot, and a robot already
has a base class it wants.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np


class HoloBaseKit:
    """Spec fixups for a planar (holonomic-joint) mobile base."""

    @staticmethod
    def add_base_sites(spec, prefix: str, base_body: str, world_z: float = 0.005) -> None:
        """The two sites ``HoloJointsRobotBaseGroup`` resolves by name.

        ``world`` is the fixed reference the base pose is expressed against and
        ``base_site`` rides the chassis body.
        """
        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, world_z], quat=[1, 0, 0, 0])
        spec.body(f"{prefix}{base_body}").add_site(
            name=f"{prefix}base_site", pos=[0, 0, 0], quat=[1, 0, 0, 0]
        )

    @staticmethod
    def add_planar_actuators(
        spec,
        prefix: str,
        joints: Sequence[str],
        act_names: Sequence[str] | None = None,
        xy_range: tuple[float, float] = (-50.0, 50.0),
        theta_range: tuple[float, float] = (-3.14159, 3.14159),
        kp: float = 15000.0,
        kv: float = 500.0,
        force: float | None = None,
        torque: float | None = None,
    ) -> list[str]:
        """Position actuators for the (x, y, theta) joints, in that order.

        ``force``/``torque`` set a symmetric force limit -- leave them None for
        an unlimited drive. A limit is usually what you want: a base servo strong
        enough to hold position under ``mj_forward`` will bury itself in a wall
        rather than be stopped by one, and once buried the robot freezes with a
        perfectly healthy planner.

        Returns the actuator names it created.
        """
        assert len(joints) == 3, f"expected (x, y, theta) joints, got {joints}"
        names = (
            list(act_names)
            if act_names is not None
            else [f"{j.removesuffix('_joint')}_act" for j in joints]
        )
        assert len(names) == 3, f"expected 3 actuator names, got {names}"
        # Non-degenerate, checked here rather than trusted: a [0, 0] range is the
        # silent drive-to-origin above, and it is easier to typo than to notice.
        for rng in (xy_range, theta_range):
            assert float(rng[1]) > float(rng[0]), f"degenerate ctrlrange {rng}"
        ranges = (xy_range, xy_range, theta_range)
        limits = (force, force, torque)
        for joint, name, rng, lim in zip(joints, names, ranges, limits, strict=True):
            a = spec.add_actuator()
            a.name = f"{prefix}{name}"
            a.target = f"{prefix}{joint}"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_position(kp, kv)
            a.ctrllimited = True
            a.ctrlrange = np.asarray(rng, float)
            if lim is not None:
                a.forcelimited = True
                a.forcerange = np.array([-float(lim), float(lim)], float)
        return [f"{prefix}{n}" for n in names]

    @staticmethod
    def add_head_camera(
        spec,
        prefix: str,
        parent_body: str,
        name: str,
        pos: Sequence[float],
        quat: Sequence[float],
        fovy: float,
    ) -> None:
        """A camera on ``parent_body``. ``quat`` is wxyz in MuJoCo's camera
        convention: the camera looks along -z with +y up, so any image ROLL of
        the real sensor mount belongs inside this quaternion."""
        cam = spec.body(f"{prefix}{parent_body}").add_camera()
        cam.name = f"{prefix}{name}"
        cam.fovy = float(fovy)
        cam.pos = np.asarray(pos, float)
        cam.quat = np.asarray(quat, float)

    @staticmethod
    def assert_planar_ctrlrange(model, actuator_names: Sequence[str]) -> None:
        """Fail at construction on a degenerate ctrlrange, instead of driving to
        the origin for a whole episode.

        Call this from the base move group's ``__init__``: by then the spec is
        compiled, so this catches a model that ships its own planar actuators
        without ranges as well as one built by ``add_planar_actuators``.
        """
        for n in actuator_names:
            a = model.actuator(n)
            lo, hi = (float(x) for x in a.ctrlrange)
            assert a.ctrllimited and hi > lo, (
                f"actuator {n!r} has a degenerate ctrlrange {(lo, hi)}: "
                f"JointPosController clips every command into it, so the base "
                f"would be driven to {lo} on the first step and held there. "
                f"Give the planar actuators a real range."
            )
