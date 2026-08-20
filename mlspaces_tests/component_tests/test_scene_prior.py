"""ScenePrior: what an agent is GIVEN at t=0, and what it is deliberately not.

The correctness property that matters is the withholding one: a category
excluded from the prior must be absent from the SEMANTICS and from the
GEOMETRY. Leaking it through geometry is the quiet failure -- a solid box at the
target's true pose is the answer, in a shape nothing else explains.
"""

import mujoco
import numpy as np
import pytest

from molmo_spaces.env.scene_prior import (
    export_scene_prior,
    normalize_category,
    surface_points,
)

# --------------------------------------------------------------------------- #
# A two-object scene: a "counter" (kept) and a "mug" (withheld), far apart.
# --------------------------------------------------------------------------- #
COUNTER_XY, MUG_XY = (0.0, 0.0), (5.0, 5.0)


def _model():
    spec = mujoco.MjSpec()
    for name, (x, y) in (("counter", COUNTER_XY), ("mug", MUG_XY)):
        body = spec.worldbody.add_body(name=name, pos=[x, y, 0.5])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.25, 0.25])
    # a child body, to pin descendant traversal
    parent = spec.worldbody.add_body(name="shelf", pos=[-3.0, 0.0, 0.5])
    child = parent.add_body(name="shelf/mesh_holder", pos=[0, 0, 0])
    child.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class _Obj:
    def __init__(self, name):
        self.name = name


class _OM:
    """Stub ObjectManager: names are their own categories; "mug" is movable."""

    def __init__(self, names):
        self._names = list(names)

    def get_objects_of_type(self, _types):
        return [_Obj(n) for n in self._names]

    def fallback_expression(self, name):
        return name

    def has_free_joint(self, name):
        return name == "mug"

    def is_structural(self, name):
        return name in ("counter", "shelf")

    def is_receptacle(self, name):
        return name == "counter"


class _Env:
    current_batch_index = 0
    current_model_path = "test_scene.xml"

    def __init__(self, model, data, om):
        self.current_model, self.current_data = model, data
        self.object_managers = [om]


def _env():
    m, d = _model()
    return _Env(m, d, _OM(["counter", "mug", "shelf"]))


def test_normalize_category_is_spacing_and_case_insensitive():
    assert normalize_category("Wine Bottle") == normalize_category("winebottle")
    assert normalize_category("Salt-Shaker") == "saltshaker"


def test_withholding_removes_the_category_from_semantics_and_geometry():
    env = _env()
    prior = export_scene_prior(
        env,
        geometry="surface_points",
        semantics="all",
        withhold_categories=["mug"],
        withhold_movable=False,
        voxel=0.1,
    )
    assert "mug" not in prior.vocabulary()
    assert "mug" in prior.withheld_categories
    assert prior.withheld_bodies, "the withheld body id must be recorded"

    # THE leak test: no prior geometry anywhere near the withheld object.
    mug = np.array([*MUG_XY, 0.5])
    d = np.linalg.norm(prior.geometry.points - mug[None, :], axis=1).min()
    assert d > 0.5, f"geometry leaks the withheld object ({d:.3f} m away)"

    # the kept object IS represented
    counter = np.array([*COUNTER_XY, 0.5])
    assert np.linalg.norm(prior.geometry.points - counter[None, :], axis=1).min() < 0.5


def test_withhold_movable_catches_free_jointed_objects():
    env = _env()
    prior = export_scene_prior(env, semantics="all", withhold_movable=True)
    assert "mug" not in prior.vocabulary()
    assert "counter" in prior.vocabulary()


def test_structural_filter():
    env = _env()
    all_objs = export_scene_prior(env, semantics="all", withhold_movable=False)
    structural = export_scene_prior(env, semantics="structural", withhold_movable=False)
    assert "mug" in all_objs.vocabulary()
    assert "mug" not in structural.vocabulary()
    assert "counter" in structural.vocabulary()


def test_geometry_and_semantics_are_independent():
    env = _env()
    sem_only = export_scene_prior(env, geometry="none", semantics="all")
    geo_only = export_scene_prior(env, geometry="surface_points", semantics="none")
    assert sem_only.geometry is None and sem_only.semantics
    assert geo_only.geometry is not None and not geo_only.semantics
    # "given nothing" is a valid control arm, not an error
    nothing = export_scene_prior(env, geometry="none", semantics="none")
    assert nothing.geometry is None and nothing.semantics == []


def test_point_clouds_are_non_degenerate():
    """A single-geom object must not collapse to one point: downstream
    point-cloud libraries cannot build geometry from a degenerate set."""
    env = _env()
    prior = export_scene_prior(env, semantics="all", withhold_movable=False)
    for obj in prior.semantics:
        assert len(obj.points) >= 8, (obj.class_name, obj.points.shape)
        assert np.ptp(obj.points, axis=0).min() > 0, "cloud is coplanar/degenerate"


def test_surface_points_skips_descendants_of_skipped_bodies():
    m, d = _model()
    shelf = m.body("shelf").id
    everything = surface_points(m, d, 0.1)
    without = surface_points(m, d, 0.1, skip_bodies=[shelf])
    assert len(without) < len(everything)
    # the shelf's geom hangs off a CHILD body; skipping the parent must drop it
    shelf_xy = np.array([-3.0, 0.0, 0.5])
    assert np.linalg.norm(without - shelf_xy[None, :], axis=1).min() > 0.5


def test_surface_points_spacing_respects_voxel():
    m, d = _model()
    coarse = surface_points(m, d, 0.5)
    fine = surface_points(m, d, 0.05)
    assert len(fine) > len(coarse)


def test_bundle_records_its_own_provenance():
    env = _env()
    prior = export_scene_prior(
        env,
        geometry="surface_points",
        semantics="all",
        withhold_categories=["mug"],
        privilege="privileged",
    )
    assert prior.protocol == 1
    assert prior.privilege == "privileged"
    assert prior.provenance["geometry"] == "surface_points"
    assert prior.provenance["semantics"] == "all"
    assert "test_scene.xml" in prior.provenance["scene"]
    assert "ScenePrior" in prior.summary()


def test_as_point_clouds_shape():
    env = _env()
    prior = export_scene_prior(env, semantics="all", withhold_movable=False)
    clouds = prior.as_point_clouds()
    assert clouds and all({"class_name", "points"} == set(c) for c in clouds)
    assert all(np.asarray(c["points"]).shape[1] == 3 for c in clouds)


@pytest.mark.parametrize("kind", ["none", "surface_points"])
def test_geometry_kinds_are_accepted(kind):
    export_scene_prior(_env(), geometry=kind)
