"""Pre-compile fixups for tasks whose correctness depends on segmentation.

MuJoCo's offscreen renderer defaults to ``visual/quality/offsamples = 4`` (MSAA).
For colour rendering that is what you want. For SEGMENTATION it is a bug: the
renderer draws each body in a unique id-colour, and MSAA *averages* neighbouring
colours at silhouette edges -- so the blended pixel decodes to a THIRD, unrelated
body id.

Measured on FloorPlan1 (RBY1 head camera): 3-5 bodies (refrigerator, toaster,
tap, window, knob) appear at 1-3 px with ``offsamples=4`` and are completely
absent at ``offsamples=0``.

Why this matters beyond cosmetics: ``NavToObjTask.check_object_visible`` is
``check_visibility(...) > 0.0`` -- ANY non-zero pixel fraction. A one-pixel
phantom is therefore a scoring event, both for policies that use segmentation as
a detector and for the task's own success test.

``offsamples`` is read by ``mjr_makeContext`` when the render context is built,
so it has to be set on the MjSpec BEFORE compilation. ``add_auxiliary_objects``
is the sanctioned pre-compile hook, which is why this is a task-sampler mixin.
"""

import logging

from mujoco import MjSpec

log = logging.getLogger(__name__)


class SegmentationRigMixin:
    """Force MSAA off so segmentation id-colours decode exactly.

    Mix in BEFORE the task-sampler base class so this ``add_auxiliary_objects``
    runs and chains upward::

        class MyTaskSampler(SegmentationRigMixin, BaseMujocoTaskSampler):
            ...

    Set ``force_no_msaa = False`` on a subclass to restore MuJoCo's default --
    useful only for measuring the phantom rate, not for producing numbers.
    """

    force_no_msaa: bool = True

    def add_auxiliary_objects(self, spec: MjSpec | None) -> None:
        super().add_auxiliary_objects(spec)
        if spec is None or not self.force_no_msaa:
            return
        if spec.visual.quality.offsamples != 0:
            log.debug(
                "segmentation rig: offsamples %d -> 0 (MSAA blends id-colors)",
                spec.visual.quality.offsamples,
            )
        spec.visual.quality.offsamples = 0
