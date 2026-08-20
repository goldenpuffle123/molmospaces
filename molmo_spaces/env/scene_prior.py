"""Export what a robot is GIVEN about a scene before it starts: the prior map.

A lot of experiments need to hand an agent some knowledge at t=0 -- "the robot
mapped this house yesterday" -- and then measure what that knowledge buys. Doing
that honestly needs three things that are easy to get wrong:

1. GEOMETRY that is neither too coarse nor too sparse (see `surface_points`).
2. SEMANTICS in the scene's own category vocabulary, so a downstream perception
   stack can treat a given object like a detected one.
3. A WITHHOLDING policy, applied to BOTH of the above and RECORDED. A prior that
   includes the task's target is not a prior, it is the answer -- and it leaks
   through geometry just as surely as through class names (a solid box at the
   target's true pose is a target-shaped hole nothing else explains).

Everything here is a read of the compiled model plus the ObjectManager. Nothing
is written back into the scene, and no task state is touched, so it is equally
usable outside an episode for offline scene analysis.

PRIVILEGE. A scene prior is privileged information by construction. The bundle
carries `privilege` and `withheld_categories` so a run can record what its agent
was given, and a reviewer can check it without rerunning anything.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

log = logging.getLogger(__name__)

PROTOCOL = 1

# 8 corners per geom rather than its centre. A one-geom object would otherwise be
# a single point, which downstream point-cloud libraries cannot build geometry
# from (open3d raises "Plane vertices are not coplanar" on a degenerate set), and
# corners also give the object the ground footprint a priority map projects.
_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
)


def normalize_category(s: Any) -> str:
    """Category names compared spacing- and case-insensitively.

    MolmoSpaces concatenates ("winebottle") where vocabularies tend to space
    ("wine bottle"); withholding has to survive that difference or it silently
    fails open.
    """
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class SemanticObject:
    """One object the agent is given."""

    name: str  # MuJoCo body name
    class_name: str  # scene category, e.g. "countertop"
    natural_name: str  # ObjectManager's phrasing, e.g. "wine bottle"
    position: np.ndarray  # [3] world
    points: np.ndarray  # [N, 3] world surface samples ([] if not requested)
    is_structural: bool
    is_receptacle: bool
    movable: bool


@dataclass
class GeometryPrior:
    """Occupied-surface geometry. `points` are world coordinates, spaced <= voxel."""

    kind: Literal["surface_points"]
    voxel: float
    points: np.ndarray  # [N, 3]


@dataclass
class ScenePrior:
    """What the agent is given, plus the record of what it was NOT given."""

    protocol: int
    privilege: str
    geometry: GeometryPrior | None
    semantics: list[SemanticObject]
    withheld_categories: list[str]
    withheld_bodies: list[int]
    provenance: dict = field(default_factory=dict)

    def vocabulary(self) -> list[str]:
        """Category names an external detector's label space must contain for
        these objects to be legal members of it (class ids, display colours and
        relevancy lookups all key off the class list)."""
        return sorted({o.class_name for o in self.semantics})

    def as_point_clouds(self) -> list[dict]:
        """`[{class_name, points}]` -- the shape semantic-mapping stacks accept
        when seeding a map with given objects."""
        return [
            {"class_name": o.class_name, "points": o.points}
            for o in self.semantics
            if len(o.points)
        ]

    def summary(self) -> str:
        n_pts = 0 if self.geometry is None else len(self.geometry.points)
        return (
            f"ScenePrior(privilege={self.privilege}, {len(self.semantics)} objects, "
            f"{n_pts} geometry points, withheld {len(self.withheld_categories)} "
            f"categories / {len(self.withheld_bodies)} bodies)"
        )


def _descendant_body(model, geom_id: int, targets: set[int]) -> int | None:
    """The ancestor of this geom's body that is in `targets`, if any.

    Needed because a ProcTHOR object is a parent body whose meshes hang off
    CHILD bodies -- reading body_geomnum on the top-level body finds nothing.
    """
    b = int(model.geom_bodyid[geom_id])
    while b > 0 and b not in targets:
        b = int(model.body_parentid[b])
    return b if b in targets else None


def _triangle_points(tri: np.ndarray, voxel: float) -> np.ndarray:
    """Barycentric lattice over triangles [n, 3, 3] with spacing <= voxel.

    The subdivision order is per triangle, from its LONGEST edge, so a 7 m wall
    quad and a 1 cm mesh detail each cost what they should. Triangles are grouped
    by order so a whole house is a few vectorized passes, not a per-face loop.
    """
    e = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(1)
    k = np.maximum(np.ceil(e / voxel).astype(int), 1)
    out = []
    for kk in np.unique(k):
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        ok = (i + j) <= kk
        w = np.stack([1 - (i[ok] + j[ok]) / kk, i[ok] / kk, j[ok] / kk], 1)
        out.append(np.einsum("pv,nvc->npc", w, tri[k == kk]).reshape(-1, 3))
    return np.vstack(out)


def surface_points(model, data, voxel: float, skip_bodies: Collection[int] = ()) -> np.ndarray:
    """World points covering every geom's SURFACE at <= voxel spacing.

    Pure MuJoCo, so it can also be run against a bare scene XML with no task.

    MESHES ARE SAMPLED ON THEIR TRIANGLES; PRIMITIVES ARE FILLED FROM THEIR AABB.
    Both halves of that are measured, and the obvious alternatives are worse:

    * Mesh VERTICES are far too sparse. A 7.0 x 0.005 x 2.6 m wall slab carries
      16 vertices, so a vertex prior leaves a wall an agent drives through.
    * A mesh's AABB is far too coarse. One wall mesh can span BOTH sides of a
      doorway plus the lintel above it, so filling its box SEALS THE DOOR --
      measured on a ProcTHOR house, filling boxes broke the interior into
      disconnected rooms (largest connected floor region 26%).

    Sampling surfaces also leaves the inside of a solid mesh unobserved, which is
    exactly what a depth sensor would leave there.

    FREE SPACE IS DELIBERATELY NOT PRODUCED. This returns occupied surfaces only.
    Marking the interior free would collapse exploration gain and change what a
    search benchmark measures; everything not returned here stays unknown.

    Args:
        skip_bodies: bodies to omit, WITH their descendants -- the robot itself,
            and whatever the withholding policy excludes.
    """
    skip = {int(b) for b in skip_bodies}
    out = []
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue  # the ground plane's AABB is the whole world
        if _descendant_body(model, g, skip) is not None:
            continue
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        mesh_id = int(model.geom_dataid[g])
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0:
            va, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            v = np.asarray(model.mesh_vert[va : va + nv], float).reshape(-1, 3)
            f = np.asarray(model.mesh_face[fa : fa + nf]).reshape(-1, 3)
            out.append(_triangle_points(v[f] @ R.T + data.geom_xpos[g], voxel))
        else:
            c = np.asarray(data.geom_xpos[g], float) + R @ model.geom_aabb[g, :3]
            h = np.abs(R) @ np.asarray(model.geom_aabb[g, 3:], float)
            ax = [
                np.linspace(c[i] - h[i], c[i] + h[i], max(int(np.ceil(2 * h[i] / voxel)) + 1, 1))
                for i in range(3)
            ]
            out.append(np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3))
    return np.vstack(out) if out else np.zeros((0, 3))


def _object_points(model, data, body_ids: dict[int, str], min_r: float) -> dict[str, list]:
    """Corner clouds per object body, gathering geoms from descendants too."""
    pts: dict[str, list] = {}
    for g in range(model.ngeom):
        b = _descendant_body(model, g, set(body_ids))
        if b is None:
            continue
        r = max(float(np.max(model.geom_size[g])), min_r)
        pts.setdefault(body_ids[b], []).extend(data.geom_xpos[g] + r * _CORNERS)
    return pts


def export_scene_prior(
    env,
    *,
    geometry: Literal["none", "surface_points"] = "none",
    semantics: Literal["none", "structural", "all"] = "none",
    withhold_categories: Collection[str] = (),
    withhold_movable: bool = True,
    voxel: float = 0.1,
    object_points: bool = True,
    skip_bodies: Collection[int] = (),
    min_corner_r: float = 0.02,
    privilege: str = "privileged",
    batch_index: int | None = None,
) -> ScenePrior:
    """Build the prior map an agent is given at t=0.

    Args:
        geometry: "surface_points" to include occupied surfaces (see
            `surface_points`), "none" for semantics only.
        semantics: "all" for every object, "structural" for fixtures only
            (walls, counters -- what a robot could plausibly have mapped and
            what stays put), "none" for geometry only.
        withhold_categories: category names to exclude, matched
            spacing-insensitively. THE TASK'S TARGET CATEGORY BELONGS HERE. The
            exclusion applies to semantics AND geometry, and is recorded on the
            bundle.
        withhold_movable: also exclude anything with a free joint. Small movable
            objects are exactly what a day-old map would have wrong, and they are
            usually the search targets.
        voxel: geometry sample spacing [m].
        object_points: attach per-object corner clouds (needed to seed a semantic
            mapper; set False if you only want poses and metadata).
        skip_bodies: extra bodies to omit from geometry, e.g. the robot's own.
        privilege: recorded on the bundle, for the run's output.
        batch_index: which env in the batch; defaults to the current one.

    Returns:
        A `ScenePrior`. Both halves may be empty -- that is a valid "given
        nothing" control arm.
    """
    idx = env.current_batch_index if batch_index is None else batch_index
    model, data = env.current_model, env.current_data
    om = env.object_managers[idx]

    exclude = {normalize_category(c) for c in withhold_categories}
    withheld_names: list[str] = []
    withheld_bodies: set[int] = set()
    kept: dict[str, tuple[str, str]] = {}  # body name -> (class_name, natural_name)

    for obj in om.get_objects_of_type([]):
        name = getattr(obj, "name", "")
        if not name:
            continue
        try:
            natural = str(om.fallback_expression(name)).lower()
        except Exception:  # noqa: BLE001 - an object without a usable name is skipped
            continue
        category = normalize_category(natural)
        drop = category in exclude
        if not drop and withhold_movable:
            try:
                drop = bool(om.has_free_joint(name))
            except Exception:  # noqa: BLE001
                drop = False
        if drop:
            withheld_names.append(natural)
            with contextlib.suppress(Exception):
                withheld_bodies.add(int(model.body(name).id))
            continue
        kept[name] = (natural, natural)

    # --- semantics -------------------------------------------------------- #
    objects: list[SemanticObject] = []
    if semantics != "none":
        ids = {}
        for nm in kept:
            with contextlib.suppress(Exception):
                ids[int(model.body(nm).id)] = nm
        clouds = _object_points(model, data, ids, min_corner_r) if object_points else {}
        for bid, nm in ids.items():
            try:
                structural = bool(om.is_structural(nm))
                receptacle = bool(om.is_receptacle(nm))
                movable = bool(om.has_free_joint(nm))
            except Exception:  # noqa: BLE001
                structural, receptacle, movable = False, False, False
            if semantics == "structural" and not structural:
                continue
            cls, natural = kept[nm]
            objects.append(
                SemanticObject(
                    name=nm,
                    class_name=cls,
                    natural_name=natural,
                    position=np.asarray(data.xpos[bid], float).copy(),
                    points=np.asarray(clouds.get(nm, []), float).reshape(-1, 3),
                    is_structural=structural,
                    is_receptacle=receptacle,
                    movable=movable,
                )
            )

    # --- geometry --------------------------------------------------------- #
    geom = None
    if geometry == "surface_points":
        pts = surface_points(model, data, voxel, skip_bodies=set(skip_bodies) | withheld_bodies)
        geom = GeometryPrior(kind="surface_points", voxel=voxel, points=pts)

    prior = ScenePrior(
        protocol=PROTOCOL,
        privilege=privilege,
        geometry=geom,
        semantics=objects,
        withheld_categories=sorted(set(withheld_names)),
        withheld_bodies=sorted(withheld_bodies),
        provenance={
            "scene": str(getattr(env, "current_model_path", "")),
            "geometry": geometry,
            "semantics": semantics,
            "voxel": voxel,
            "withhold_movable": withhold_movable,
        },
    )
    log.info("%s", prior.summary())
    return prior
