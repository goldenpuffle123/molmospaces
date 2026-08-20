"""Task progress with explicit privilege tiers.

A policy driving a nav task cannot see why it is not succeeding. The grader's
test lives in ``NavToObjTask.get_reward`` and combines two conditions measured a
specific way -- 2D distance from the robot base to the target's BODY ORIGIN,
against ``succ_pos_threshold``, AND non-zero head-camera visibility -- and the
only feedback a policy gets is the boolean. Every stack that wants to know "am I
close, and is it visible" therefore re-derives the test, and any disagreement
about what the distance is measured to becomes a silent loss.

This exposes the answer, and stamps WHO IS ALLOWED TO KNOW IT into the payload,
because the interesting failure here is silent privilege leakage rather than a
missing number:

* ``JUDGE``        - exactly what the grader computes. Read-only, always safe to
                     record; whether it may be a policy INPUT is a leaderboard
                     rule, which is why the tier is recorded rather than assumed.
* ``SHAPING``      - dense privileged signal (geodesic distance, visibility
                     fraction). Fine for training and diagnosis, taints a
                     benchmark number.
* ``UNPRIVILEGED`` - what a deployed robot could compute itself: its own pose,
                     elapsed time, remaining budget.

Requesting a method above the oracle's tier raises. That is the mechanism a
comment cannot be: ``progress()["privilege"]`` travels with the numbers, so a
run recorded at JUDGE can be rejected by one grep instead of by trust.
"""

from __future__ import annotations

import heapq
import logging
from enum import StrEnum
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PROTOCOL = 1


class Privilege(StrEnum):
    """Who is allowed to know what. Ordered by RANK, see ``_RANK``."""

    UNPRIVILEGED = "unprivileged"
    SHAPING = "shaping"
    JUDGE = "judge"


_RANK = {Privilege.UNPRIVILEGED: 0, Privilege.SHAPING: 1, Privilege.JUDGE: 2}


class TaskOracle:
    """Tier-filtered progress for one environment of a nav task.

    Args:
        task: the running task (``NavToObjTask``).
        privilege: the highest tier this oracle will answer.
        batch_index: which environment; None = the env's current batch index.
        dt_s: seconds per POLICY step, used by ``elapsed_s``. The task counts
            steps, not time, and the decision period is the caller's own
            property -- a deployed robot knows it, so it is unprivileged.
        geodesic_res: cell size [m] for the flood behind
            ``geodesic_distance_to_target``.
    """

    def __init__(
        self,
        task,
        privilege: Privilege = Privilege.UNPRIVILEGED,
        batch_index: int | None = None,
        dt_s: float = 0.2,
        geodesic_res: float = 0.05,
    ) -> None:
        self.task = task
        self.privilege = Privilege(privilege)
        self._batch_index = batch_index
        self.dt_s = float(dt_s)
        self.geodesic_res = float(geodesic_res)
        self._field: tuple[Any, ...] | None = None  # (key, dist, s, k, h, w)

    # ---- plumbing ------------------------------------------------------- #
    @property
    def index(self) -> int:
        return self.task.env.current_batch_index if self._batch_index is None else self._batch_index

    def _require(self, tier: Privilege, what: str) -> None:
        if _RANK[self.privilege] < _RANK[tier]:
            raise PermissionError(
                f"{what} is {tier.value}-tier; this oracle is "
                f"{self.privilege.value}. Construct it with "
                f"privilege=Privilege.{tier.name} if the experiment allows it."
            )

    def _target(self):
        return self.task.get_nearest_nav_object(self.index)

    def _base_xy(self) -> np.ndarray:
        pose = self.task.env.robots[self.index].robot_view.base.pose
        return np.asarray(pose[:3, 3], float)[:2]

    # ---- UNPRIVILEGED --------------------------------------------------- #
    def steps_remaining(self) -> int:
        """Steps left in the budget, or -1 when the horizon is unbounded."""
        horizon = getattr(self.task, "_task_horizon", np.inf)
        if not np.isfinite(horizon):
            return -1
        return int(max(0, horizon - self.task.episode_step_count))

    def elapsed_s(self) -> float:
        return float(self.task.episode_step_count * self.dt_s)

    # ---- SHAPING -------------------------------------------------------- #
    def distance_to_target(self) -> float:
        """Straight-line 2D distance to the target's body origin -- the SAME
        quantity and the same convention the grader thresholds."""
        self._require(Privilege.SHAPING, "distance_to_target")
        return float(
            np.linalg.norm(np.asarray(self._target().position, float)[:2] - self._base_xy())
        )

    def visibility_fraction(self) -> float:
        """Fraction of the head-camera image the target occupies. The grader
        only asks whether this is > 0."""
        self._require(Privilege.SHAPING, "visibility_fraction")
        return float(self.task.env.check_visibility("head_camera", self._target().name))

    def geodesic_distance_to_target(self) -> float:
        """Walkable distance [m] over the placement occupancy map, or inf when
        the target's neighbourhood is unreachable from the robot.

        The target itself sits INSIDE the dilated obstacle set (it is furniture),
        so the flood is seeded from every free cell within the success threshold
        of it, each at its own straight-line offset -- i.e. this measures the
        walk to the success annulus, which is the distance that can actually be
        driven. One flood per episode, cached."""
        self._require(Privilege.SHAPING, "geodesic_distance_to_target")
        field = self._distance_field()
        if field is None:
            return float("inf")
        dist, lo, res = field
        r, c = np.floor((self._base_xy() - lo) / res).astype(int)
        h, w = dist.shape
        if not (0 <= r < h and 0 <= c < w):
            return float("inf")
        # SNAP. The placement map is dilated by the agent radius, so a robot
        # hugging a wall stands in a cell that map calls occupied and the flood
        # never reached -- reporting inf there would read as "unreachable" when
        # the truth is "half a body-radius off the walkable set". Take the best
        # finite value within one agent radius instead, and only give up if the
        # whole neighbourhood is unreached.
        if not np.isfinite(dist[r, c]):
            k = max(1, int(round(0.35 / res)))
            win = dist[max(0, r - k) : r + k + 1, max(0, c - k) : c + k + 1]
            return float(win.min())
        return float(dist[r, c])

    def _distance_field(self):
        """One flood per (scene, target), cached. Returns (dist, lo, res) or None.

        Built from ``thormap.get_free_points()`` -- the map's OWN answer to
        "where can the robot stand" -- rather than from ``occupancy_map`` plus
        ``occupancy_scale_factor``. Two reasons, both measured:
        ``ProcTHORMap.__init__`` never passes ``occupancy_scale_factor`` to its
        base, so that attribute is None and the inherited pixel->world
        ``__call__`` raises on any ProcTHOR scene; and ProcTHORMap stores
        occupancy with NON-ZERO MEANING FREE (see get_free_points), the opposite
        of the usual reading. get_free_points sidesteps both and is the same
        function robot placement trusts.
        """
        env = self.task.env
        target = np.asarray(self._target().position, float)[:2]
        key = (getattr(env, "current_model_path", ""), tuple(np.round(target, 3)))
        if self._field is not None and self._field[0] == key:
            return self._field[1]

        try:
            pts = np.asarray(self.task.env.get_thormap().get_free_points(), float)
        except Exception as e:  # noqa: BLE001 - no placement map on this env
            # inf rather than a crash, but SAID once: a silent inf reads exactly
            # like "unreachable", and those are different facts.
            log.warning(
                "no occupancy map for the geodesic distance (%s: %s); "
                "geodesic_distance_to_target will report inf",
                type(e).__name__,
                e,
            )
            self._field = (key, None)
            return None
        if pts.size == 0:
            self._field = (key, None)
            return None

        res = self.geodesic_res
        lo = pts[:, :2].min(axis=0) - res
        idx = np.floor((pts[:, :2] - lo) / res).astype(int)
        shape = tuple(idx.max(axis=0) + 2)
        free = np.zeros(shape, bool)
        free[idx[:, 0], idx[:, 1]] = True

        # The target is furniture: it sits in the occupied set, so the flood is
        # seeded from every free cell within the success threshold, each at its
        # own straight-line offset. What this then measures is the walk to the
        # success annulus -- the distance that can actually be driven.
        thr = float(self.task.config.task_config.succ_pos_threshold)
        rr, cc = np.nonzero(free)
        off = np.hypot(lo[0] + (rr + 0.5) * res - target[0], lo[1] + (cc + 0.5) * res - target[1])
        seed = off <= thr
        if not seed.any():
            seed = off == off.min()

        dist = np.full(free.shape, np.inf)
        heap = []
        for r, c, d in zip(rr[seed], cc[seed], off[seed], strict=True):
            dist[r, c] = d
            heap.append((float(d), int(r), int(c)))
        heapq.heapify(heap)
        diag = res * np.sqrt(2.0)
        nbrs = (
            (-1, 0, res),
            (1, 0, res),
            (0, -1, res),
            (0, 1, res),
            (-1, -1, diag),
            (-1, 1, diag),
            (1, -1, diag),
            (1, 1, diag),
        )
        h, w = free.shape
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, cost in nbrs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and free[nr, nc]:
                    nd = d + cost
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(heap, (nd, nr, nc))
        self._field = (key, (dist, lo, res))
        return self._field[1]

    # ---- JUDGE ---------------------------------------------------------- #
    def would_succeed_now(self) -> bool:
        """The grader's test, evaluated right now."""
        self._require(Privilege.JUDGE, "would_succeed_now")
        return bool(self.task.get_reward()[self.index] > 0.0)

    def success_breakdown(self) -> dict:
        """WHY the grader's test passes or fails, split into its two halves.

        ``measured_to`` is stated because it is the half that surprises people:
        the distance is to the target's BODY ORIGIN, not to its nearest surface,
        so a large object can be touched and still be out of range."""
        self._require(Privilege.JUDGE, "success_breakdown")
        target = self._target()
        d = float(np.linalg.norm(np.asarray(target.position, float)[:2] - self._base_xy()))
        thr = float(self.task.config.task_config.succ_pos_threshold)
        vis = float(self.task.env.check_visibility("head_camera", target.name))
        return {
            "success": bool(d < thr and vis > 0.0),
            "within_radius": bool(d < thr),
            "visible": bool(vis > 0.0),
            "distance_m": d,
            "threshold_m": thr,
            "visibility_fraction": vis,
            "measured_to": "body_origin",
            "target": target.name,
        }

    # ---- the tier-filtered view ----------------------------------------- #
    def progress(self) -> dict:
        """Everything this oracle's tier allows, stamped with that tier."""
        out: dict[str, Any] = {
            "protocol": PROTOCOL,
            "privilege": self.privilege.value,
            "steps_remaining": self.steps_remaining(),
            "elapsed_s": self.elapsed_s(),
            "base_xy": self._base_xy().tolist(),
        }
        if _RANK[self.privilege] >= _RANK[Privilege.SHAPING]:
            out["distance_m"] = self.distance_to_target()
            out["geodesic_m"] = self.geodesic_distance_to_target()
            out["visibility_fraction"] = self.visibility_fraction()
        if _RANK[self.privilege] >= _RANK[Privilege.JUDGE]:
            out["success_breakdown"] = self.success_breakdown()
            out["would_succeed_now"] = out["success_breakdown"]["success"]
        return out
