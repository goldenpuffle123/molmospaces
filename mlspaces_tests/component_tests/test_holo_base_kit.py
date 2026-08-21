"""The three attach-time fixups every mobile MJCF needed.

The one that matters is ctrlrange: MuJoCo defaults it to [0, 0] and
JointPosController clips every command into it, so a model that merely forgets
the range drives its base to the world origin and holds it there -- a silent
failure that reads as a planner bug. These pin the fixups and that assert.
"""

import mujoco
import numpy as np
import pytest

from molmo_spaces.robots.holo_base_kit import HoloBaseKit

BASE_XML = """
<mujoco model="planar">
  <worldbody>
    <body name="x_prismatic_link" pos="0 0 0">
      <joint name="x_to_world_joint" type="slide" axis="1 0 0"/>
      <!-- the virtual links need mass: MuJoCo refuses to compile a moving body
           without inertia once it is driven -->
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <body name="y_prismatic_link">
        <joint name="y_to_x_joint" type="slide" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <body name="base_link">
          <joint name="base_to_y_joint" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.3 0.2 0.1"/>
          <body name="head_link" pos="0 0 0.5">
            <geom type="sphere" size="0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

JOINTS = ("x_to_world_joint", "y_to_x_joint", "base_to_y_joint")
ACTS = ("x_to_world_act", "y_to_x_act", "base_to_y_act")


def _spec(prefix: str = ""):
    """A planar-base model. `prefix` mimics an attached robot, whose bodies and
    joints are already namespaced by the time the kit runs."""
    xml = BASE_XML
    for name in ("x_prismatic_link", "y_prismatic_link", "base_link", "head_link", *JOINTS):
        xml = xml.replace(f'"{name}"', f'"{prefix}{name}"')
    return mujoco.MjSpec.from_string(xml)


def test_base_sites_are_named_where_the_base_group_looks():
    spec = _spec("robot_0/")
    HoloBaseKit.add_base_sites(spec, "robot_0/", "base_link")
    model = spec.compile()
    assert model.site("robot_0/world").id >= 0
    assert model.site("robot_0/base_site").id >= 0
    # base_site rides the chassis, world does not.
    assert model.site("robot_0/base_site").bodyid[0] == model.body("robot_0/base_link").id


def test_planar_actuators_get_real_ranges_and_force_limits():
    spec = _spec("robot_0/")
    names = HoloBaseKit.add_planar_actuators(
        spec,
        "robot_0/",
        JOINTS,
        ACTS,
        xy_range=(-100.0, 100.0),
        theta_range=(-3.14, 3.14),
        kp=5000.0,
        kv=500.0,
        force=100.0,
        torque=20.0,
    )
    assert names == [f"robot_0/{a}" for a in ACTS]
    model = spec.compile()
    for name, rng in zip(names, ((-100, 100), (-100, 100), (-3.14, 3.14)), strict=True):
        a = model.actuator(name)
        assert bool(a.ctrllimited)
        np.testing.assert_allclose(a.ctrlrange, rng, atol=1e-6)
    np.testing.assert_allclose(model.actuator(names[0]).forcerange, (-100, 100))
    np.testing.assert_allclose(model.actuator(names[2]).forcerange, (-20, 20))


def test_force_limit_is_optional():
    spec = _spec()
    names = HoloBaseKit.add_planar_actuators(spec, "", JOINTS, ACTS)
    model = spec.compile()
    assert not bool(model.actuator(names[0]).forcelimited)


def test_degenerate_range_is_refused_at_build_time():
    spec = _spec()
    with pytest.raises(AssertionError, match="degenerate ctrlrange"):
        HoloBaseKit.add_planar_actuators(spec, "", JOINTS, ACTS, xy_range=(0.0, 0.0))


def test_assert_planar_ctrlrange_catches_a_model_that_shipped_its_own():
    """The failure this exists for: actuators present, ranges absent."""
    spec = _spec()
    for joint, name in zip(JOINTS, ACTS, strict=True):
        a = spec.add_actuator()
        a.name = name
        a.target = joint
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.set_to_position(5000.0, 500.0)
    model = spec.compile()
    with pytest.raises(AssertionError, match="degenerate ctrlrange"):
        HoloBaseKit.assert_planar_ctrlrange(model, ACTS)


def test_assert_planar_ctrlrange_passes_on_a_kit_built_base():
    spec = _spec()
    names = HoloBaseKit.add_planar_actuators(spec, "", JOINTS, ACTS)
    HoloBaseKit.assert_planar_ctrlrange(spec.compile(), names)


def test_head_camera_lands_on_its_parent_with_the_given_optics():
    spec = _spec("robot_0/")
    HoloBaseKit.add_head_camera(
        spec, "robot_0/", "head_link", "head_cam", [0.03, 0.0, 0.03], [0.5, -0.5, 0.5, 0.5], 87.0
    )
    model = spec.compile()
    cam = model.camera("robot_0/head_cam")
    assert cam.bodyid[0] == model.body("robot_0/head_link").id
    assert float(cam.fovy[0]) == pytest.approx(87.0)
    np.testing.assert_allclose(cam.pos, [0.03, 0.0, 0.03], atol=1e-9)


def test_three_joints_are_required():
    spec = _spec()
    with pytest.raises(AssertionError, match="expected .x, y, theta."):
        HoloBaseKit.add_planar_actuators(spec, "", JOINTS[:2], ACTS[:2])
