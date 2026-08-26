"""Configs for the external-module bridge (``molmo_spaces/bridge``).

``BridgePolicyConfig`` is the whole knob surface on THIS side of the wire: where
the external endpoints are, which sensor streams to render for them, which
privilege tier their progress payload may carry, and an opaque ``params`` dict
forwarded verbatim. Everything about HOW the far side plans stays over there.

``BridgeNavToObjConfig`` is a runnable object-goal-navigation experiment driven
that way -- the ~10-line example an external stack copies, and what
``mlspaces_tests`` uses as the conformance target:

    # terminal 1 (their env)   python my_stack.py --port 8800
    # terminal 2 (this env)    python molmo_spaces/data_generation/main.py \\
    #                              molmo_spaces.configs.bridge_configs:BridgeNavToObjConfig
"""

from __future__ import annotations

import os
from typing import Literal

from molmo_spaces.bridge.policy import BridgePolicy
from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, RBY1Config
from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler


class BridgePolicyConfig(BasePolicyConfig):
    """Where the external stack is, and what it is allowed to see."""

    policy_cls: type = BridgePolicy
    policy_factory: object = BridgePolicy
    policy_type: str = "bridge"

    # --- endpoints. Modules run FIRST, in list order, each returning an
    # observation augmentation the next endpoint sees; the policy answers last.
    policy_uri: str = os.environ.get("MS_BRIDGE_POLICY", "ws://127.0.0.1:8800")
    module_uris: list[str] = []
    connection_timeout: float | None = None  # None = wait forever for the peer

    # --- what to render for them. The subscription in the reset REPLY decides
    # what is actually sent; this decides what exists to send.
    cameras: list[str] = ["head_camera"]
    depth: bool = True
    segmentation: bool = True
    self_mask: bool = True
    camera_params: bool = False  # nav's own sensor bundle already attaches these

    # --- what they are told
    dt_s: float = 0.2  # == policy_dt_ms; the far side's control period
    # None = report whatever the ROBOT declares (BaseRobotConfig.base_kinematics),
    # which is the only place that fact is decided. Setting it here overrides that
    # for the wire, which is wanted only when an experiment deliberately drives a
    # robot under a motion model other than its own -- and then the episode
    # payload records the model the run actually claimed.
    base_kinematics: str | None = None
    base_action_kind: str = "planar_pose"
    params: dict = {}  # opaque, forwarded verbatim in episode["params"]
    episode_extra: dict = {}

    # --- privilege. progress_every=0 never sends it; the JUDGE tier renders a
    # segmentation frame per call, so the cadence is a real cost knob.
    oracle_privilege: str = "judge"
    progress_every: int = 25

    # --- optional prior map handed over at t=0 (env/scene_prior.py)
    prior_geometry: Literal["none", "surface_points"] = "none"
    prior_semantics: Literal["none", "all", "structural"] = "none"
    prior_voxel: float = 0.1
    prior_withhold_categories: list[str] = []
    prior_withhold_target: bool = True  # never hand over the episode's own target
    prior_withhold_movable: bool = False


@register_config("BridgeNavToObjConfig")
class BridgeNavToObjConfig(NavToObjBaseConfig):
    """Object-goal navigation, decided by an external process."""

    task_type: str = "nav_to_obj"
    robot_config: BaseRobotConfig = RBY1Config()
    policy_config: BasePolicyConfig = BridgePolicyConfig()
    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        # NavToObjTaskSampler already mixes in SegmentationRigMixin (MSAA off), so
        # a client computing its own target mask from segmentation ids is safe --
        # and episode["scene"]["offsamples"] states the value so it can assert it.
        task_sampler_class=NavToObjTaskSampler,
        filter_for_successful_trajectories=False,
    )
    num_envs: int = 1
    filter_for_successful_trajectories: bool = False
    end_on_success: bool = True

    def _init_policy_config(self) -> BasePolicyConfig:
        return self.policy_config

    @property
    def tag(self) -> str:
        return "bridge_nav_to_obj"
