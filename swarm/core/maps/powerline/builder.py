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

"""Type 9 power lines: a peri-urban road corridor with an 11 kV feeder, an LT line and service drops.

The hazard is the wire, not the pole. Conductors are drawn at their real 3-6 mm radius and
collide at WIRE_COLLISION_RADIUS_MIN_M, so a depth camera rarely sees them while the physics
always does; swarm.core.tof_wire_model decides per frame whether a ToF sensor would have.
Wires come in every height band a drone crosses: 11 kV phases at 9-11 m, LT at 6-7 m, service
drops sagging to 3-4 m at the huts, and stay wires running diagonally to the ground.

The corridor is built in a local frame (lines along +x) and turned by a seeded yaw. Line offsets
are drawn so that no line crosses a clearing; a span that still would is dropped.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from swarm.core.maps.biome_kit import MAP_HALF_EXTENT_M, Frame, Kit, WireSpan
from swarm.core.scenario import scenario_for_seed

POWERLINE_CHALLENGE_TYPE = 9
POWERLINE_SEED_OFFSET = 0x9A11

GROUND_RGBA_BY_SEASON: Dict[str, Tuple[float, float, float, float]] = {
    "summer": (0.64, 0.56, 0.42, 1.0),
    "monsoon": (0.38, 0.45, 0.27, 1.0),
    "post_monsoon": (0.46, 0.45, 0.31, 1.0),
    "winter": (0.48, 0.46, 0.36, 1.0),
    "snow": (0.90, 0.91, 0.93, 1.0),
}
ROAD_RGBA = (0.24, 0.24, 0.25, 1.0)
CONCRETE_RGBA = (0.72, 0.71, 0.68, 1.0)
STEEL_RGBA = (0.55, 0.57, 0.58, 1.0)
CONDUCTOR_RGBA = (0.42, 0.43, 0.44, 1.0)   # bare aluminium, weathered
LT_WIRE_RGBA = (0.15, 0.15, 0.15, 1.0)      # aerial-bunched / insulated
TRUNK_RGBA = (0.36, 0.27, 0.18, 1.0)
LEAF_RGBA = (0.22, 0.40, 0.18, 1.0)
WALL_RGBAS = ((0.80, 0.74, 0.62, 1.0), (0.70, 0.78, 0.80, 1.0), (0.86, 0.80, 0.70, 1.0))
ROOF_RGBA = (0.52, 0.30, 0.22, 1.0)

# Real conductor radii (m): ACSR "Rabbit" on 11 kV, LT phase wire, service drop, stay strand, HT phase.
R_11KV, R_LT, R_SERVICE, R_STAY, R_HT = 0.005, 0.004, 0.003, 0.004, 0.012
LINE_CLEAR_MARGIN_M = 2.0
OFFSET_TRIES = 30
HT_CORRIDOR_Y = 55.0  # the transmission line runs past the playable +-42 m, as a backdrop and a ceiling


def _line_is_clear(kit: Kit, f: Frame, ly: float) -> bool:
    """True when the full-length line at local offset ly misses every clearing."""
    return kit.segment_clear(f.w(-MAP_HALF_EXTENT_M, ly), f.w(MAP_HALF_EXTENT_M, ly), margin=LINE_CLEAR_MARGIN_M)


def _pick_offset(kit: Kit, rng: random.Random, f: Frame, lo: float, hi: float) -> float:
    """A line offset in [lo, hi] that clears the clearings, else the last draw (its spans get dropped)."""
    ly = rng.uniform(lo, hi)
    for _ in range(OFFSET_TRIES):
        if _line_is_clear(kit, f, ly):
            return ly
        ly = rng.uniform(lo, hi)
    return ly


def _pole_xs(rng: random.Random, span: Tuple[float, float]) -> List[float]:
    xs, x = [], -MAP_HALF_EXTENT_M + rng.uniform(0.0, 8.0)
    while x <= MAP_HALF_EXTENT_M:
        xs.append(x)
        x += rng.uniform(*span)
    return xs


def _stay(kit: Kit, f: Frame, lx: float, ly: float, top_z: float, outward: float) -> int:
    """A stay wire from near the pole top to a ground anchor beyond the line end."""
    anchor = (lx + outward * 0.45 * top_z, ly)
    if not kit.segment_clear(f.w(lx, ly), f.w(*anchor), margin=1.0):
        return 0
    kit.wire(WireSpan(f.w3(lx, ly, top_z - 0.4), f.w3(anchor[0], anchor[1], 0.05), 0.0, R_STAY, "stay"),
             STEEL_RGBA, n_segments=3)
    return 1


def _strung_line(kit: Kit, rng: random.Random, f: Frame, ly: float, poles_x: Sequence[float], pole_h: float,
                 phases: Sequence[Tuple[float, float]], radius: float, sag: Tuple[float, float], kind: str,
                 rgba: Sequence[float], pole_r: float) -> Tuple[List[float], int]:
    """Poles plus conductors at (across-offset, drop-below-top) per phase; returns kept poles and wire count."""
    kept = [lx for lx in poles_x if kit.clear(*f.w(lx, ly), 1.0)]
    arm_half = max(abs(a) for a, _ in phases) + 0.15
    for lx in kept:
        kit.cylinder(*f.w(lx, ly), 0.0, pole_r, pole_h, CONCRETE_RGBA)
        kit.box(f.w3(lx, ly, pole_h - 0.25), (0.05, arm_half, 0.05), STEEL_RGBA, yaw=f.yaw)  # crossarm
    n = 0
    for a, b in zip(kept, kept[1:]):
        if not kit.segment_clear(f.w(a, ly), f.w(b, ly), margin=1.0):
            continue
        for across, drop in phases:
            z = pole_h - drop
            kit.wire(WireSpan(f.w3(a, ly + across, z), f.w3(b, ly + across, z), rng.uniform(*sag), radius, kind), rgba)
            n += 1
    if len(kept) >= 2:
        n += _stay(kit, f, kept[0], ly, pole_h, -1.0) + _stay(kit, f, kept[-1], ly, pole_h, 1.0)
    return kept, n


def _ht_tower_line(kit: Kit, rng: random.Random, f: Frame, side: float) -> int:
    """A lattice-tower transmission line beyond the playable range: 3 phases at 14-22 m plus an earth wire."""
    ly = side * HT_CORRIDOR_Y
    top = rng.uniform(20.0, 25.0)
    xs = [-50.0, 0.0, 50.0]
    for lx in xs:
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):  # four legs leaning in to a waist
            kit.rod(f.w3(lx + dx * 1.6, ly + dy * 1.6, 0.0), f.w3(lx + dx * 0.4, ly + dy * 0.4, top - 4.0),
                    0.08, STEEL_RGBA)
        for k, z in enumerate((top - 6.0, top - 3.0)):
            kit.box(f.w3(lx, ly, z), (0.15, 4.0 - k, 0.12), STEEL_RGBA, yaw=f.yaw)
        kit.rod(f.w3(lx, ly, top - 6.0), f.w3(lx, ly, top), 0.12, STEEL_RGBA)
    n = 0
    phases = [(-3.8, top - 6.0, "ht"), (3.8, top - 6.0, "ht"), (2.8, top - 3.0, "ht"), (0.0, top, "earth")]
    for a, b in zip(xs, xs[1:]):
        for across, z, kind in phases:
            kit.wire(WireSpan(f.w3(a, ly + across, z), f.w3(b, ly + across, z), rng.uniform(1.5, 2.5), R_HT, kind),
                     CONDUCTOR_RGBA, n_segments=10)
            n += 1
    return n


def _huts_and_drops(kit: Kit, rng: random.Random, f: Frame, lt_y: float, lt_poles: Sequence[float],
                    lt_top: float, side: float) -> Tuple[int, int]:
    """Huts set back from the LT line, each fed by a sagging service drop from the nearest pole."""
    huts = drops = 0
    for _ in range(rng.randint(4, 8)):
        lx = rng.uniform(-48.0, 48.0)
        ly = lt_y + side * rng.uniform(6.0, 12.0)
        x, y = f.w(lx, ly)
        if not kit.clear(x, y, 4.5):
            continue
        h = rng.uniform(2.8, 3.6)
        kit.hut(x, y, rng.uniform(3.5, 5.0), rng.uniform(3.0, 4.5), h, rng.choice(WALL_RGBAS), ROOF_RGBA, yaw=f.yaw)
        huts += 1
        if not lt_poles:
            continue
        px = min(lt_poles, key=lambda q: abs(q - lx))
        if abs(px - lx) > 25.0 or not kit.segment_clear(f.w(px, lt_y), (x, y), margin=1.0):
            continue
        kit.wire(WireSpan(f.w3(px, lt_y, lt_top - 1.0), (x, y, h + 0.4), rng.uniform(0.3, 0.6), R_SERVICE, "service"),
                 LT_WIRE_RGBA)
        drops += 1
    return huts, drops


def _roadside_trees(kit: Kit, rng: random.Random, f: Frame, line_ys: Sequence[float], season: str) -> int:
    """Neem and peepal along the road, kept off the line corridors (where real crews lop them)."""
    crown = "none" if season == "snow" else "round"
    n = 0
    for _ in range(rng.randint(10, 20)):
        lx, ly = rng.uniform(-55.0, 55.0), rng.uniform(-45.0, 45.0)
        if any(abs(ly - y) < 4.0 for y in line_ys):
            continue
        x, y = f.w(lx, ly)
        if kit.clear(x, y, 3.5):
            kit.tree(x, y, rng.uniform(6.0, 11.0), rng.uniform(2.0, 3.5), TRUNK_RGBA, LEAF_RGBA, crown=crown)
            n += 1
    return n


def build_powerline_map(cli: int, seed: int, safe_zones, safe_zone_radius: float) -> dict:
    """Build the type 9 power-line corridor for a seed; returns the asset dict the map dispatcher tags."""
    rng = random.Random((int(seed) + POWERLINE_SEED_OFFSET) & 0xFFFFFFFF)
    season = scenario_for_seed(int(seed), POWERLINE_CHALLENGE_TYPE).season
    f = Frame(rng.uniform(0.0, math.pi))

    kit = Kit(cli, safe_zones, safe_zone_radius)
    kit.ground(GROUND_RGBA_BY_SEASON[season])

    hv_y = _pick_offset(kit, rng, f, -30.0, 30.0)
    side = rng.choice((-1.0, 1.0))
    lo, hi = sorted((hv_y + side * 10.0, hv_y + side * 18.0))
    lt_y = _pick_offset(kit, rng, f, lo, hi)
    road_half_w = min(0.5 * abs(lt_y - hv_y) - 2.0, 3.5)
    if road_half_w > 1.5:  # drawn only: the floor slab under it is what a crash hits
        kit.box(f.w3(0.0, (hv_y + lt_y) / 2.0, 0.01), (MAP_HALF_EXTENT_M, road_half_w, 0.01), ROAD_RGBA,
                yaw=f.yaw, collide=False, bucket="props")

    hv_h = rng.uniform(10.0, 11.0)
    _, n_hv = _strung_line(kit, rng, f, hv_y, _pole_xs(rng, (40.0, 50.0)), hv_h,
                           [(-0.6, 0.3), (0.6, 0.3), (0.0, -0.4)], R_11KV, (0.6, 1.0), "11kv", CONDUCTOR_RGBA, 0.14)
    lt_h = rng.uniform(7.5, 8.5)
    lt_poles, n_lt = _strung_line(kit, rng, f, lt_y, _pole_xs(rng, (30.0, 36.0)), lt_h,
                                  [(0.0, 1.0), (0.0, 1.4), (0.0, 1.8), (0.0, 2.2)], R_LT, (0.4, 0.8), "lt",
                                  LT_WIRE_RGBA, 0.11)
    huts, n_drops = _huts_and_drops(kit, rng, f, lt_y, lt_poles, lt_h, side)
    n_trees = _roadside_trees(kit, rng, f, (hv_y, lt_y), season)
    n_ht = _ht_tower_line(kit, rng, f, -side) if rng.random() < 0.5 else 0

    return kit.summary(biome="powerline", season=season, yaw=round(f.yaw, 3), hv_wires=n_hv, lt_wires=n_lt,
                       service_drops=n_drops, ht_wires=n_ht, huts=huts, roadside_trees=n_trees)


__all__ = ["POWERLINE_CHALLENGE_TYPE", "build_powerline_map"]
