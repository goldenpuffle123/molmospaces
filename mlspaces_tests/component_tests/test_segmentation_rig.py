"""MSAA must be off wherever segmentation id-colors are decoded.

MuJoCo defaults visual/quality/offsamples to 4, which averages id-colors at
silhouette edges into unrelated body ids. Nav success is "any non-zero pixel
fraction", so a blended phantom is a scoring event.
"""

import mujoco

from molmo_spaces.tasks.segmentation_rig import SegmentationRigMixin


class _Base:
    """Stands in for BaseMujocoTaskSampler: records that super() was chained."""

    def __init__(self):
        self.base_called = False

    def add_auxiliary_objects(self, spec):
        self.base_called = True


class _Sampler(SegmentationRigMixin, _Base):
    pass


class _OptedOut(_Sampler):
    force_no_msaa = False


def test_mujoco_default_is_the_hazard():
    """Guard the premise: if MuJoCo ever defaults to 0, this mixin is dead code."""
    assert mujoco.MjSpec().visual.quality.offsamples == 4


def test_mixin_forces_offsamples_to_zero():
    spec = mujoco.MjSpec()
    sampler = _Sampler()
    sampler.add_auxiliary_objects(spec)
    assert spec.visual.quality.offsamples == 0
    assert sampler.base_called, "must chain to super().add_auxiliary_objects"


def test_survives_compilation():
    """offsamples is read when the render context is built, so it must survive
    compile() -- setting it on the spec post-compile would be a no-op."""
    spec = mujoco.MjSpec()
    _Sampler().add_auxiliary_objects(spec)
    model = spec.compile()
    assert model.vis.quality.offsamples == 0


def test_none_spec_is_tolerated():
    """add_auxiliary_objects is called with None on some paths."""
    sampler = _Sampler()
    sampler.add_auxiliary_objects(None)
    assert sampler.base_called


def test_opt_out_restores_the_default():
    spec = mujoco.MjSpec()
    _OptedOut().add_auxiliary_objects(spec)
    assert spec.visual.quality.offsamples == 4


def test_nav_sampler_inherits_the_mixin():
    from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler

    assert issubclass(NavToObjTaskSampler, SegmentationRigMixin)
    # The mixin must precede the base so its add_auxiliary_objects wins.
    mro = NavToObjTaskSampler.__mro__
    from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler

    assert mro.index(SegmentationRigMixin) < mro.index(BaseMujocoTaskSampler)
