# The MIT License (MIT)
# Copyright © 2026 Swarm
# Copyright © 2026 Nirmitk Tripathii (biome and scenario extensions)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Type 10 terraced hills: a Himalayan valley side cut into stepped fields, with stone huts and conifers.

The hill rises along a seeded axis from a flat valley floor. Each tread is a solid step 1.0-1.8 m
above the one below and 4-8 m deep; the riser lines wander along the contour (a sine, broken into
6-10 m segments) the way hand-built terraces do. One tread is one compound body, so a twenty-step
hill costs the placement raycasts twenty bodies.

Pads on this map are placed by surface raycast (like the mountain), which puts a pad at the highest
ground under its footprint. A pad straddling a riser would float above the lower tread, so every
clearing gets a level notch cut into the hillside at the tread height under its centre.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from swarm.core.maps.biome_kit import MAP_HALF_EXTENT_M, Frame, Kit, tint
from swarm.core.scenario import scenario_for_seed

TERRACED_HILLS_CHALLENGE_TYPE = 10
TERRACED_HILLS_SEED_OFFSET = 0x7E44

RISER_M = (1.0, 1.8)
TREAD_M = (4.0, 8.0)
SEGMENT_M = (6.0, 10.0)
CONTOUR_AMP_M = (1.0, 3.0)
CONTOUR_WAVELENGTH_M = (35.0, 70.0)
VALLEY_EDGE_M = (-55.0, -25.0)   # local u where the first riser rises from the valley floor
BUILD_HALF_M = MAP_HALF_EXTENT_M + 5.0
NOTCH_HALF_M = 3.5               # level cut around each pad: wider than the finder's 1.15 x pad radius probe
MIN_RECT_M = 0.05
RISER_KEEPOUT_M = 1.0            # trees and huts stay this far off a riser line

TREAD_RGBA_BY_SEASON: Dict[str, Tuple[float, float, float, float]] = {
    "summer": (0.60, 0.50, 0.36, 1.0),        # ploughed, before the rains
    "monsoon": (0.30, 0.50, 0.22, 1.0),       # paddy and millet in leaf
    "post_monsoon": (0.66, 0.60, 0.30, 1.0),  # ripe rice
    "winter": (0.46, 0.50, 0.28, 1.0),        # wheat and barley shoots
    "snow": (0.92, 0.93, 0.95, 1.0),
}
VALLEY_RGBA_BY_SEASON: Dict[str, Tuple[float, float, float, float]] = {
    "summer": (0.55, 0.50, 0.40, 1.0),
    "monsoon": (0.32, 0.44, 0.26, 1.0),
    "post_monsoon": (0.44, 0.46, 0.32, 1.0),
    "winter": (0.48, 0.47, 0.38, 1.0),
    "snow": (0.88, 0.89, 0.92, 1.0),
}
LANDING_RGBA = (0.58, 0.52, 0.42, 1.0)       # beaten-earth threshing floor
STREAM_RGBA = (0.32, 0.45, 0.52, 1.0)
ICE_RGBA = (0.78, 0.84, 0.88, 1.0)
STONE_RGBA = (0.62, 0.58, 0.52, 1.0)
SLATE_RGBA = (0.32, 0.33, 0.36, 1.0)
TRUNK_RGBA = (0.34, 0.25, 0.17, 1.0)
CONIFER_RGBA = (0.16, 0.30, 0.18, 1.0)
CONIFER_SNOW_RGBA = (0.62, 0.70, 0.66, 1.0)
BROADLEAF_RGBA = (0.26, 0.42, 0.20, 1.0)

Rect = Tuple[float, float, float, float]  # (u0, u1, v0, v1) in the local frame


@dataclass(frozen=True)
class TerracePlan:
    """Pure terrace geometry: riser lines along u, contour segments along v, tread tops in z."""

    edges: Tuple[float, ...]      # u of each riser before the contour offset; tread k spans edges[k]..edges[k+1]
    tops: Tuple[float, ...]       # top z of tread k, strictly increasing
    seg_edges: Tuple[float, ...]  # v boundaries of the contour segments
    amp: float
    wavelength: float
    phase: float

    @property
    def n_treads(self) -> int:
        return len(self.tops)

    def segment(self, v: float) -> int:
        j = bisect.bisect_right(self.seg_edges, v) - 1
        return min(max(j, 0), len(self.seg_edges) - 2)

    def offset(self, j: int) -> float:
        """Contour shift of segment j: constant along the segment, so each step is a box."""
        vc = 0.5 * (self.seg_edges[j] + self.seg_edges[j + 1])
        return self.amp * math.sin(2.0 * math.pi * vc / self.wavelength + self.phase)

    def tread_at(self, u: float, v: float) -> int:
        """Tread index under (u, v); -1 on the valley floor."""
        uo = u - self.offset(self.segment(v))
        return min(bisect.bisect_right(self.edges, uo) - 1, self.n_treads - 1)

    def surface_z(self, u: float, v: float) -> float:
        k = self.tread_at(u, v)
        return 0.0 if k < 0 else self.tops[k]

    def riser_distance(self, u: float, v: float) -> float:
        """Distance along u to the nearest riser line."""
        uo = u - self.offset(self.segment(v))
        return min(abs(uo - e) for e in self.edges)


def plan_terraces(rng: random.Random) -> TerracePlan:
    """Draw a terrace layout that covers the map from the valley edge to beyond the far side."""
    u = rng.uniform(*VALLEY_EDGE_M)
    edges, tops, z = [u], [], 0.0
    amp = rng.uniform(*CONTOUR_AMP_M)
    while u < BUILD_HALF_M + amp:
        z += rng.uniform(*RISER_M)
        tops.append(round(z, 4))
        u += rng.uniform(*TREAD_M)
        edges.append(u)
    v, segs = -BUILD_HALF_M, [-BUILD_HALF_M]
    while v < BUILD_HALF_M:
        v = min(v + rng.uniform(*SEGMENT_M), BUILD_HALF_M)
        segs.append(v)
    return TerracePlan(tuple(edges), tuple(tops), tuple(segs), amp, rng.uniform(*CONTOUR_WAVELENGTH_M),
                       rng.uniform(0.0, 2.0 * math.pi))


def cut_square(rect: Rect, cu: float, cv: float, half: float) -> List[Rect]:
    """rect minus the square centred (cu, cv) with half-side `half`, as up to four rects."""
    u0, u1, v0, v1 = rect
    su0, su1, sv0, sv1 = cu - half, cu + half, cv - half, cv + half
    if u1 <= su0 or u0 >= su1 or v1 <= sv0 or v0 >= sv1:
        return [rect]
    out = []
    if v0 < sv0:
        out.append((u0, u1, v0, sv0))
    if v1 > sv1:
        out.append((u0, u1, sv1, v1))
    mv0, mv1 = max(v0, sv0), min(v1, sv1)
    if u0 < su0:
        out.append((u0, su0, mv0, mv1))
    if u1 > su1:
        out.append((su1, u1, mv0, mv1))
    return [r for r in out if r[1] - r[0] > MIN_RECT_M and r[3] - r[2] > MIN_RECT_M]


def tread_rects(plan: TerracePlan, k: int, notches: Sequence[Tuple[float, float]]) -> List[Rect]:
    """Footprint rects of tread k, one per contour segment, with every notch square cut out."""
    rects: List[Rect] = []
    for j in range(len(plan.seg_edges) - 1):
        off = plan.offset(j)
        u0 = max(plan.edges[k] + off, -BUILD_HALF_M)
        u1 = min(plan.edges[k + 1] + off, BUILD_HALF_M)
        if u1 - u0 <= MIN_RECT_M:
            continue
        pieces = [(u0, u1, plan.seg_edges[j], plan.seg_edges[j + 1])]
        for cu, cv in notches:
            pieces = [r for piece in pieces for r in cut_square(piece, cu, cv, NOTCH_HALF_M)]
        rects.extend(pieces)
    return rects


def _build_treads(kit: Kit, f: Frame, plan: TerracePlan, notches: Sequence[Tuple[float, float]],
                  rgba: Sequence[float]) -> int:
    """Each tread as one compound body of solid steps from just below the floor to its top."""
    n = 0
    for k, top in enumerate(plan.tops):
        parts = []
        for u0, u1, v0, v1 in tread_rects(plan, k, notches):
            parts.append((((u0 + u1) / 2.0, (v0 + v1) / 2.0, (top - 0.1) / 2.0),
                          ((u1 - u0) / 2.0, (v1 - v0) / 2.0, (top + 0.1) / 2.0)))
        shade = 1.0 if k % 2 == 0 else 0.93  # alternate treads read as separate steps from the air
        if kit.boxes(parts, tint(rgba, shade), yaw=f.yaw, bucket="terrain") is not None:
            n += 1
    return n


def _build_landings(kit: Kit, f: Frame, plan: TerracePlan, notches: Sequence[Tuple[float, float]]) -> int:
    """A level floor in each notch at the tread height under its centre (the valley floor needs none)."""
    n = 0
    for cu, cv in notches:
        z = plan.surface_z(cu, cv)
        if z <= 0.0:
            continue
        kit.box(f.w3(cu, cv, (z - 0.1) / 2.0), (NOTCH_HALF_M, NOTCH_HALF_M, (z + 0.1) / 2.0), LANDING_RGBA,
                yaw=f.yaw, bucket="terrain")
        n += 1
    return n


def _build_stream(kit: Kit, rng: random.Random, f: Frame, plan: TerracePlan, season: str) -> None:
    """A drawn-only stream on the valley floor below the first riser."""
    u = plan.edges[0] - rng.uniform(4.0, 10.0)
    if u - 2.0 < -BUILD_HALF_M:
        return
    half_w = rng.uniform(1.0, 2.0)
    kit.box(f.w3(u, 0.0, 0.015), (half_w, BUILD_HALF_M, 0.015), ICE_RGBA if season == "snow" else STREAM_RGBA,
            yaw=f.yaw, collide=False, bucket="props")


def _build_huts(kit: Kit, rng: random.Random, f: Frame, plan: TerracePlan) -> int:
    """Stone huts with slate roofs, each set on one tread clear of its risers."""
    n = 0
    for _ in range(30):
        if n >= 6:
            break
        k = rng.randrange(plan.n_treads)
        j = rng.randrange(len(plan.seg_edges) - 1)
        depth_avail = plan.edges[k + 1] - plan.edges[k] - 2.0 * RISER_KEEPOUT_M
        d = rng.uniform(3.0, 4.5)
        if d > depth_avail:
            continue
        u = 0.5 * (plan.edges[k] + plan.edges[k + 1]) + plan.offset(j)
        v = 0.5 * (plan.seg_edges[j] + plan.seg_edges[j + 1])
        if abs(u) > MAP_HALF_EXTENT_M - 5.0 or abs(v) > MAP_HALF_EXTENT_M - 5.0:
            continue
        w = rng.uniform(3.5, 5.0)
        x, y = f.w(u, v)
        if not kit.clear(x, y, 0.5 * max(w, d) + 1.0):
            continue
        kit.hut(x, y, d, w, rng.uniform(2.6, 3.2), STONE_RGBA, SLATE_RGBA, yaw=f.yaw, z0=plan.tops[k])
        n += 1
    return n


def _build_trees(kit: Kit, rng: random.Random, f: Frame, plan: TerracePlan, season: str) -> int:
    """Conifers on the upper slope, walnut and apple lower down and in the valley; leaf-off under snow."""
    z_max = plan.tops[-1]
    n = 0
    for _ in range(rng.randint(30, 45)):
        u, v = rng.uniform(-55.0, 55.0), rng.uniform(-55.0, 55.0)
        if plan.riser_distance(u, v) < RISER_KEEPOUT_M:
            continue
        z0 = plan.surface_z(u, v)
        conifer = z0 > 0.4 * z_max
        crown_r = rng.uniform(1.4, 2.2) if conifer else rng.uniform(2.0, 3.0)
        x, y = f.w(u, v)
        if not kit.clear(x, y, crown_r):
            continue
        if conifer:
            leaf = CONIFER_SNOW_RGBA if season == "snow" else CONIFER_RGBA
            kit.tree(x, y, rng.uniform(9.0, 15.0), crown_r, TRUNK_RGBA, leaf, crown="cone", z0=z0)
        else:
            crown = "none" if season == "snow" else "round"
            kit.tree(x, y, rng.uniform(6.0, 9.0), crown_r, TRUNK_RGBA, BROADLEAF_RGBA, crown=crown, z0=z0)
        n += 1
    return n


def build_terraced_hills_map(cli: int, seed: int, safe_zones, safe_zone_radius: float) -> dict:
    """Build the type 10 terraced hillside for a seed; returns the asset dict the map dispatcher tags."""
    rng = random.Random((int(seed) + TERRACED_HILLS_SEED_OFFSET) & 0xFFFFFFFF)
    season = scenario_for_seed(int(seed), TERRACED_HILLS_CHALLENGE_TYPE).season
    f = Frame(rng.uniform(0.0, 2.0 * math.pi))
    plan = plan_terraces(rng)
    notches = [f.local(x, y) for x, y in safe_zones]

    kit = Kit(cli, safe_zones, safe_zone_radius)
    kit.ground(VALLEY_RGBA_BY_SEASON[season])
    n_treads = _build_treads(kit, f, plan, notches, TREAD_RGBA_BY_SEASON[season])
    n_landings = _build_landings(kit, f, plan, notches)
    _build_stream(kit, rng, f, plan, season)
    huts = _build_huts(kit, rng, f, plan)
    trees = _build_trees(kit, rng, f, plan, season)

    return kit.summary(biome="terraced_hills", season=season, yaw=round(f.yaw, 3), treads=n_treads,
                       top_m=round(plan.tops[-1], 2), landings=n_landings, huts=huts, trees=trees)


__all__ = ["TERRACED_HILLS_CHALLENGE_TYPE", "TerracePlan", "build_terraced_hills_map", "cut_square",
           "plan_terraces", "tread_rects"]
