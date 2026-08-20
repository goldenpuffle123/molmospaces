"""Task progress with privilege tiers.

The grader's test lives inside NavToObjTask and a policy only ever sees the
boolean, so every stack re-derives it and can disagree about what the distance
is measured to. TaskOracle exposes the breakdown and, more importantly, stamps
the tier that was available -- these tests pin the gating and the stamp.
"""

import numpy as np
import pytest

from molmo_spaces.env.task_oracle import PROTOCOL, Privilege, TaskOracle


class _Obj:
    def __init__(self, name, position):
        self.name = name
        self.position = np.asarray(position, float)


class _Base:
    def __init__(self, xy):
        self.pose = np.eye(4)
        self.pose[:2, 3] = xy


class _Robot:
    def __init__(self, xy):
        self.robot_view = type("_V", (), {"base": _Base(xy)})()


class _Env:
    current_batch_index = 0
    current_model_path = "stub"

    def __init__(self, xy, visibility):
        self.robots = [_Robot(xy)]
        self._vis = visibility

    def check_visibility(self, camera_name, body_name):
        return self._vis


class _Task:
    def __init__(
        self,
        robot_xy=(0.0, 0.0),
        target=(1.0, 0.0),
        visibility=0.01,
        threshold=1.5,
        steps=10,
        horizon=100,
    ):
        self.env = _Env(robot_xy, visibility)
        self._target = _Obj("mug_0", (*target, 0.8))
        self.episode_step_count = steps
        self._task_horizon = horizon
        self.config = type(
            "_C", (), {"task_config": type("_T", (), {"succ_pos_threshold": threshold})()}
        )()

    def get_nearest_nav_object(self, index):
        return self._target

    def get_reward(self):
        d = np.linalg.norm(
            self._target.position[:2] - self.env.robots[0].robot_view.base.pose[:2, 3]
        )
        thr = self.config.task_config.succ_pos_threshold
        return np.array([0.0 if self.env._vis <= 0 else max(0.0, 1.0 - d / thr)])


def test_unprivileged_tier_answers_only_its_own_questions():
    o = TaskOracle(_Task(), Privilege.UNPRIVILEGED, dt_s=0.2)
    p = o.progress()
    assert p["privilege"] == "unprivileged" and p["protocol"] == PROTOCOL
    assert p["steps_remaining"] == 90
    assert p["elapsed_s"] == pytest.approx(2.0)
    assert "distance_m" not in p and "success_breakdown" not in p
    for call in (o.distance_to_target, o.visibility_fraction, o.would_succeed_now):
        with pytest.raises(PermissionError):
            call()


def test_shaping_tier_stops_short_of_the_grader():
    o = TaskOracle(_Task(), Privilege.SHAPING)
    p = o.progress()
    assert p["privilege"] == "shaping"
    assert p["distance_m"] == pytest.approx(1.0)
    assert p["visibility_fraction"] == pytest.approx(0.01)
    assert "success_breakdown" not in p
    with pytest.raises(PermissionError):
        o.success_breakdown()


def test_judge_breakdown_splits_the_two_halves():
    o = TaskOracle(_Task(visibility=0.0), Privilege.JUDGE)
    b = o.success_breakdown()
    assert b["within_radius"] is True and b["visible"] is False
    assert b["success"] is False and o.would_succeed_now() is False
    # The half that surprises people is stated, not implied.
    assert b["measured_to"] == "body_origin"


def test_judge_agrees_with_the_task_reward():
    for xy, vis in (((0.0, 0.0), 0.01), ((5.0, 0.0), 0.01), ((0.0, 0.0), 0.0)):
        t = _Task(robot_xy=xy, visibility=vis)
        o = TaskOracle(t, Privilege.JUDGE)
        assert o.would_succeed_now() == bool(t.get_reward()[0] > 0.0)
        assert o.success_breakdown()["success"] == o.would_succeed_now()


def test_distance_is_measured_to_the_body_origin_not_the_surface():
    """A large object can be touched and still be out of range -- the asymmetry
    the breakdown exists to make visible."""
    o = TaskOracle(_Task(robot_xy=(0.0, 0.0), target=(1.6, 0.0), threshold=1.5), Privilege.JUDGE)
    b = o.success_breakdown()
    assert b["distance_m"] == pytest.approx(1.6) and b["within_radius"] is False


def test_unbounded_horizon_reports_minus_one():
    t = _Task(horizon=np.inf)
    assert TaskOracle(t).steps_remaining() == -1
