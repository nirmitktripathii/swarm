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

"""Type 8 farmland: Indian row crops, flooded paddy or an orchard, staged by the seeded season.

Geometry follows the season the scenario sampler draws for the seed, so a monsoon seed flies
over standing maize and flooded paddy while a summer seed flies over ploughed fallow. A plot
of rows is one compound body (see Kit.boxes), so a full field costs the placement probe a
handful of tests, not hundreds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from swarm.core.maps.biome_kit import (
    MAP_HALF_EXTENT_M,
    Kit,
    WireSpan,
    chord_hole,
    subtract_intervals,
    tint,
)
from swarm.core.scenario import scenario_for_seed

FARMLAND_CHALLENGE_TYPE = 8
FARMLAND_SEED_OFFSET = 0xFA4A
MODE_WEIGHTS: Dict[str, float] = {"row_crops": 0.45, "paddy": 0.30, "orchard": 0.25}
ORCHARD_HALF_EXTENT_M = 42.0

GROUND_RGBA_BY_SEASON: Dict[str, Tuple[float, float, float, float]] = {
    "summer": (0.62, 0.52, 0.38, 1.0),
    "monsoon": (0.36, 0.42, 0.24, 1.0),
    "post_monsoon": (0.45, 0.42, 0.28, 1.0),
    "winter": (0.42, 0.40, 0.30, 1.0),
    "snow": (0.90, 0.91, 0.93, 1.0),
}
BUND_RGBA = (0.50, 0.40, 0.28, 1.0)
WATER_RGBA = (0.30, 0.42, 0.48, 1.0)
TRUNK_RGBA = (0.36, 0.27, 0.18, 1.0)
LEAF_RGBA = (0.20, 0.38, 0.16, 1.0)
WALL_RGBA = (0.78, 0.72, 0.62, 1.0)
ROOF_RGBA = (0.55, 0.30, 0.22, 1.0)
POLE_RGBA = (0.70, 0.70, 0.68, 1.0)
WIRE_RGBA = (0.15, 0.15, 0.15, 1.0)


@dataclass(frozen=True)
class Crop:
    """A crop stage: canopy height range (m), row pitch (m), row half-width (m) and colour.

    A stand crop (wheat, mustard, rice) has half-width = pitch / 2, so rows touch and the
    field reads as one canopy; a row crop (maize, cotton, cane) keeps gaps between rows.
    """

    name: str
    height_m: Tuple[float, float]
    pitch_m: float
    half_width_m: float
    rgba: Tuple[float, float, float, float]


MAIZE = Crop("maize", (1.5, 2.4), 1.5, 0.25, (0.30, 0.50, 0.18, 1.0))
MAIZE_MATURE = Crop("maize_mature", (2.0, 2.6), 1.5, 0.25, (0.62, 0.58, 0.30, 1.0))
COTTON = Crop("cotton", (1.0, 1.6), 1.8, 0.35, (0.28, 0.44, 0.22, 1.0))
CANE = Crop("sugarcane", (2.5, 4.0), 2.0, 0.40, (0.36, 0.56, 0.22, 1.0))
WHEAT = Crop("wheat", (0.6, 1.0), 2.0, 1.0, (0.50, 0.58, 0.26, 1.0))
MUSTARD = Crop("mustard", (1.0, 1.6), 2.0, 1.0, (0.86, 0.78, 0.18, 1.0))
FODDER = Crop("sorghum_fodder", (0.6, 1.2), 1.5, 0.30, (0.42, 0.52, 0.24, 1.0))

ROW_CROPS_BY_SEASON: Dict[str, Sequence[Optional[Crop]]] = {
    "summer": (None, None, FODDER, CANE),               # mostly ploughed fallow before the monsoon
    "monsoon": (MAIZE, COTTON, CANE, MAIZE),            # kharif
    "post_monsoon": (MAIZE_MATURE, COTTON, CANE, None),  # harvest under way
    "winter": (WHEAT, MUSTARD, WHEAT, CANE),            # rabi
    "snow": (None, None, None, None),                   # fallow under frost
}

# Rice stage by season: (canopy height range, colour); None = no standing crop.
RICE_BY_SEASON: Dict[str, Optional[Tuple[Tuple[float, float], Tuple[float, float, float, float]]]] = {
    "summer": ((0.10, 0.20), (0.62, 0.55, 0.35, 1.0)),   # dry stubble
    "monsoon": ((0.40, 0.70), (0.34, 0.60, 0.22, 1.0)),  # transplanted, flooded
    "post_monsoon": ((0.80, 1.10), (0.74, 0.66, 0.28, 1.0)),  # ripening, golden
    "winter": ((0.10, 0.20), (0.58, 0.52, 0.34, 1.0)),
    "snow": None,
}
FLOODED_SEASONS = ("monsoon", "post_monsoon")


def _row_parts(kit: Kit, x0: float, y0: float, x1: float, y1: float, *, along_x: bool, pitch: float,
               half_w: float, height: float, z0: float = 0.0) -> List[Tuple[Tuple[float, float, float],
                                                                          Tuple[float, float, float]]]:
    """Box parts for parallel rows filling a rectangle, cut wherever a row would enter a clearing."""
    parts = []
    lo_across, hi_across = (y0, y1) if along_x else (x0, x1)
    lo_along, hi_along = (x0, x1) if along_x else (y0, y1)
    across = lo_across + pitch / 2.0
    while across <= hi_across - pitch / 2.0 + 1e-9:
        holes = []
        for sx, sy in kit.safe_zones:
            c_along, c_across = (sx, sy) if along_x else (sy, sx)
            hole = chord_hole(across, c_along, c_across, kit.clear_radius + half_w)
            if hole is not None:
                holes.append(hole)
        for a, b in subtract_intervals(lo_along, hi_along, holes, min_len=0.5):
            mid = (a + b) / 2.0
            centre = (mid, across, z0 + height / 2.0) if along_x else (across, mid, z0 + height / 2.0)
            half = ((b - a) / 2.0, half_w, height / 2.0) if along_x else (half_w, (b - a) / 2.0, height / 2.0)
            parts.append((centre, half))
        across += pitch
    return parts


def _bund_parts(kit: Kit, x0: float, y0: float, x1: float, y1: float, height: float, half_w: float):
    """The four earth bunds around a plot, cut at clearings."""
    parts = []
    for across, along_x in ((y0, True), (y1, True), (x0, False), (x1, False)):
        lo, hi = (x0, x1) if along_x else (y0, y1)
        holes = []
        for sx, sy in kit.safe_zones:
            c_along, c_across = (sx, sy) if along_x else (sy, sx)
            hole = chord_hole(across, c_along, c_across, kit.clear_radius + half_w)
            if hole is not None:
                holes.append(hole)
        for a, b in subtract_intervals(lo, hi, holes, min_len=0.5):
            mid = (a + b) / 2.0
            centre = (mid, across, height / 2.0) if along_x else (across, mid, height / 2.0)
            half = ((b - a) / 2.0, half_w, height / 2.0) if along_x else (half_w, (b - a) / 2.0, height / 2.0)
            parts.append((centre, half))
    return parts


def _plots(rng: random.Random, size_m: Tuple[float, float], track_m: float) -> List[Tuple[float, float, float, float]]:
    """Rectangular plots tiling the map, separated by cart tracks."""
    plots = []
    extent = MAP_HALF_EXTENT_M
    x = -extent
    while x < extent - 4.0:
        w = min(rng.uniform(*size_m), extent - x)
        y = -extent
        while y < extent - 4.0:
            d = min(rng.uniform(*size_m), extent - y)
            plots.append((x, y, x + w - track_m, y + d - track_m))
            y += d
        x += w
    return plots


def _build_row_crops(kit: Kit, rng: random.Random, season: str) -> Dict[str, int]:
    """Plots of row or stand crops for the season, some left fallow."""
    counts = {"plots": 0, "cropped": 0}
    choices = ROW_CROPS_BY_SEASON[season]
    for x0, y0, x1, y1 in _plots(rng, (18.0, 30.0), 3.0):
        counts["plots"] += 1
        crop = choices[rng.randrange(len(choices))]
        kit.boxes(_bund_parts(kit, x0, y0, x1, y1, 0.25, 0.2), BUND_RGBA)
        if crop is None:
            continue
        height = rng.uniform(*crop.height_m)
        parts = _row_parts(kit, x0 + 0.6, y0 + 0.6, x1 - 0.6, y1 - 0.6, along_x=rng.random() < 0.5,
                           pitch=crop.pitch_m, half_w=crop.half_width_m, height=height)
        if kit.boxes(parts, tint(crop.rgba, rng.uniform(0.9, 1.1))) is not None:
            counts["cropped"] += 1
    return counts


def _build_paddy(kit: Kit, rng: random.Random, season: str) -> Dict[str, int]:
    """Small bunded paddy plots: flooded with rice in the monsoon, stubble otherwise."""
    counts = {"plots": 0, "cropped": 0}
    rice = RICE_BY_SEASON[season]
    flooded = season in FLOODED_SEASONS
    for x0, y0, x1, y1 in _plots(rng, (12.0, 20.0), 1.2):
        counts["plots"] += 1
        kit.boxes(_bund_parts(kit, x0, y0, x1, y1, 0.35, 0.25), BUND_RGBA)
        if flooded:
            # water is only drawn: the floor slab under it is what a crash hits
            kit.slab(x0 + 0.3, y0 + 0.3, x1 - 0.3, y1 - 0.3, 0.03, 0.02, WATER_RGBA, collide=False, bucket="props")
        if rice is None:
            continue
        (h_lo, h_hi), rgba = rice
        parts = _row_parts(kit, x0 + 0.4, y0 + 0.4, x1 - 0.4, y1 - 0.4, along_x=True, pitch=1.0,
                           half_w=0.5, height=rng.uniform(h_lo, h_hi))
        if kit.boxes(parts, rgba) is not None:
            counts["cropped"] += 1
    return counts


def _build_orchard(kit: Kit, rng: random.Random, season: str) -> Dict[str, int]:
    """A jittered grid of mango/guava-like trees in a central block; leaf-off only under snow.

    The block (+-ORCHARD_HALF_EXTENT_M) holds 90-130 trees, the density of the hard forest, so
    the step cost stays in the benchmark's range; fallow ground surrounds it.
    """
    spacing = rng.uniform(7.0, 9.5)
    crown = "none" if season == "snow" else "round"
    n = 0
    extent = ORCHARD_HALF_EXTENT_M
    x = -extent
    while x <= extent:
        y = -extent
        while y <= extent:
            tx, ty = x + rng.uniform(-0.6, 0.6), y + rng.uniform(-0.6, 0.6)
            height = rng.uniform(4.0, 7.0)
            crown_r = rng.uniform(1.6, 2.6)
            if kit.clear(tx, ty, crown_r):
                kit.tree(tx, ty, height, crown_r, TRUNK_RGBA, tint(LEAF_RGBA, rng.uniform(0.85, 1.15)), crown=crown)
                n += 1
            y += spacing
        x += spacing
    return {"trees": n}


def _build_farmstead(kit: Kit, rng: random.Random) -> Optional[Tuple[float, float]]:
    """A pump house with a neem tree beside it; returns its position (for a service drop)."""
    for _ in range(40):
        x, y = rng.uniform(-45.0, 45.0), rng.uniform(-45.0, 45.0)
        if kit.clear(x, y, 7.5):
            kit.hut(x, y, 3.0, 2.5, 2.6, WALL_RGBA, ROOF_RGBA, yaw=rng.uniform(0.0, 3.14))
            kit.tree(x + 4.0, y + 1.0, rng.uniform(7.0, 10.0), rng.uniform(2.5, 3.5), TRUNK_RGBA, LEAF_RGBA)
            return x, y
    return None


def _build_lt_line(kit: Kit, rng: random.Random, farmstead: Optional[Tuple[float, float]]) -> int:
    """A low-tension line on 7-8 m poles along a track, plus a sagging service drop to the pump house.

    Spans or poles that would cross a clearing are skipped: the pads must never sit under a wire.
    """
    track_y = rng.uniform(-40.0, 40.0)
    pole_h = rng.uniform(7.0, 8.0)
    xs = []
    x = -MAP_HALF_EXTENT_M + rng.uniform(0.0, 10.0)
    while x <= MAP_HALF_EXTENT_M:
        xs.append(x)
        x += rng.uniform(30.0, 40.0)
    poles = [(px, track_y) for px in xs if kit.clear(px, track_y, 1.0)]
    for px, py in poles:
        kit.cylinder(px, py, 0.0, 0.11, pole_h, POLE_RGBA)
    n_wires = 0
    for (ax, ay), (bx, by) in zip(poles, poles[1:]):
        if not kit.segment_clear((ax, ay), (bx, by), margin=1.0):
            continue
        for dz in (0.3, 0.8):
            span = WireSpan((ax, ay, pole_h - dz), (bx, by, pole_h - dz), rng.uniform(0.4, 0.8), 0.005, "lt")
            kit.wire(span, WIRE_RGBA)
            n_wires += 1
    if poles and farmstead is not None:
        fx, fy = farmstead
        px, py = min(poles, key=lambda q: (q[0] - fx) ** 2 + (q[1] - fy) ** 2)
        if kit.segment_clear((px, py), (fx, fy), margin=1.0) and (px - fx) ** 2 + (py - fy) ** 2 > 4.0:
            drop = WireSpan((px, py, pole_h - 1.2), (fx, fy, 3.2), rng.uniform(0.3, 0.6), 0.004, "service")
            kit.wire(drop, WIRE_RGBA)
            n_wires += 1
    return n_wires


def build_farmland_map(cli: int, seed: int, safe_zones, safe_zone_radius: float) -> dict:
    """Build the type 8 farmland for a seed; returns the asset dict the map dispatcher tags."""
    rng = random.Random((int(seed) + FARMLAND_SEED_OFFSET) & 0xFFFFFFFF)
    season = scenario_for_seed(int(seed), FARMLAND_CHALLENGE_TYPE).season
    modes = list(MODE_WEIGHTS)
    mode = rng.choices(modes, weights=[MODE_WEIGHTS[m] for m in modes], k=1)[0]

    kit = Kit(cli, safe_zones, safe_zone_radius)
    kit.ground(GROUND_RGBA_BY_SEASON[season])
    if mode == "row_crops":
        counts = _build_row_crops(kit, rng, season)
    elif mode == "paddy":
        counts = _build_paddy(kit, rng, season)
    else:
        counts = _build_orchard(kit, rng, season)
    farmstead = _build_farmstead(kit, rng)
    n_wires = _build_lt_line(kit, rng, farmstead) if rng.random() < 0.6 else 0
    return kit.summary(biome="farmland", mode=mode, season=season, lt_wires=n_wires, **counts)


__all__ = ["FARMLAND_CHALLENGE_TYPE", "build_farmland_map"]
