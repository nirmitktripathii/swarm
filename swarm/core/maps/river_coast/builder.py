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

"""Type 11 river and coast: an embanked river with a road bridge, or a fishing coast with a jetty.

River mode: a channel whose width follows the season (a summer trickle between sandbars, a
monsoon flood bank to bank), levees on both banks, a road bridge on piers with its deck at
4-6 m, country boats, and sometimes an LT line strung across the water. Coast mode: sea, beach,
coconut palms, a casuarina windbreak, boats hauled up on the sand, a jetty on piles and a
lighthouse. Snow seeds always draw the river (a Himalayan river in winter); India's coasts do not
freeze.

Water is drawn only: the floor slab under it is what a crash hits, as with the forest. A depth
sensor therefore sees water as a flat floor at z = 0, which is how a ToF return off calm water
behaves at short range anyway.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from swarm.core.maps.biome_kit import (
    MAP_HALF_EXTENT_M,
    Frame,
    Kit,
    WireSpan,
    chord_hole,
    subtract_intervals,
)
from swarm.core.scenario import scenario_for_seed

RIVER_COAST_CHALLENGE_TYPE = 11
RIVER_COAST_SEED_OFFSET = 0xB1A7
COAST_PROBABILITY = 0.45
BUILD_HALF_M = MAP_HALF_EXTENT_M + 5.0
OFFSET_TRIES = 30
YAW_TRIES = 12

# River half-width (m) and water colour by season: the monsoon fills the channel bank to bank.
RIVER_BY_SEASON: Dict[str, Tuple[Tuple[float, float], Tuple[float, float, float, float]]] = {
    "summer": ((5.0, 8.0), (0.30, 0.44, 0.46, 1.0)),
    "monsoon": ((12.0, 17.0), (0.46, 0.39, 0.27, 1.0)),
    "post_monsoon": ((9.0, 13.0), (0.33, 0.42, 0.40, 1.0)),
    "winter": ((7.0, 10.0), (0.29, 0.42, 0.47, 1.0)),
    "snow": ((5.0, 8.0), (0.72, 0.79, 0.83, 1.0)),
}
SEA_RGBA_BY_SEASON: Dict[str, Tuple[float, float, float, float]] = {
    "summer": (0.20, 0.42, 0.55, 1.0),
    "monsoon": (0.30, 0.38, 0.40, 1.0),
    "post_monsoon": (0.22, 0.40, 0.50, 1.0),
    "winter": (0.24, 0.44, 0.56, 1.0),
    "snow": (0.24, 0.44, 0.56, 1.0),
}
GROUND_RGBA_BY_SEASON: Dict[str, Tuple[float, float, float, float]] = {
    "summer": (0.60, 0.53, 0.40, 1.0),
    "monsoon": (0.34, 0.44, 0.26, 1.0),
    "post_monsoon": (0.42, 0.46, 0.30, 1.0),
    "winter": (0.46, 0.46, 0.34, 1.0),
    "snow": (0.90, 0.91, 0.93, 1.0),
}
SAND_RGBA = (0.82, 0.74, 0.56, 1.0)
SURF_RGBA = (0.90, 0.93, 0.94, 1.0)
LEVEE_RGBA = (0.50, 0.44, 0.32, 1.0)
CONCRETE_RGBA = (0.68, 0.67, 0.64, 1.0)
TIMBER_RGBA = (0.42, 0.32, 0.22, 1.0)
HULL_RGBAS = ((0.20, 0.34, 0.55, 1.0), (0.62, 0.22, 0.18, 1.0), (0.40, 0.30, 0.20, 1.0), (0.18, 0.50, 0.40, 1.0))
TRUNK_RGBA = (0.36, 0.27, 0.18, 1.0)
LEAF_RGBA = (0.22, 0.42, 0.18, 1.0)
PALM_RGBA = (0.28, 0.46, 0.20, 1.0)
CASUARINA_RGBA = (0.24, 0.36, 0.22, 1.0)
WALL_RGBA = (0.80, 0.74, 0.62, 1.0)
THATCH_RGBA = (0.60, 0.50, 0.32, 1.0)
WHITE_RGBA = (0.94, 0.94, 0.92, 1.0)
RED_RGBA = (0.72, 0.16, 0.14, 1.0)
WIRE_RGBA = (0.15, 0.15, 0.15, 1.0)
LEVEE_GAP_M = 2.5      # water edge to levee centreline
LEVEE_HALF_W = 2.0


# -- shared pieces ----------------------------------------------------------------
def _water(kit: Kit, f: Frame, u0: float, u1: float, v0: float, v1: float, rgba: Sequence[float],
           z: float = 0.02) -> None:
    """A drawn-only water (or sand) sheet over the local rect; the floor slab is what collides."""
    kit.box(f.w3((u0 + u1) / 2.0, (v0 + v1) / 2.0, z - 0.01), ((u1 - u0) / 2.0, (v1 - v0) / 2.0, 0.01), rgba,
            yaw=f.yaw, collide=False, bucket="props")


def _boat(kit: Kit, rng: random.Random, f: Frame, u: float, v: float, heading: float, z0: float = 0.0) -> bool:
    """A country boat (hull plus an optional mast) if its footprint is clear of the clearings."""
    length, beam = rng.uniform(4.0, 7.5), rng.uniform(1.2, 1.9)
    x, y = f.w(u, v)
    if not kit.clear(x, y, length / 2.0):
        return False
    kit.box((x, y, z0 + 0.3), (length / 2.0, beam / 2.0, 0.3), rng.choice(HULL_RGBAS), yaw=f.yaw + heading)
    if rng.random() < 0.5:
        kit.rod((x, y, z0 + 0.6), (x, y, z0 + 0.6 + rng.uniform(3.0, 6.0)), 0.06, TIMBER_RGBA)
    return True


def _local_zones(kit: Kit, f: Frame) -> List[Tuple[float, float]]:
    return [f.local(x, y) for x, y in kit.safe_zones]


# -- river mode ---------------------------------------------------------------------
def _pick_river_axis(kit: Kit, rng: random.Random, f: Frame, band_half: float) -> float:
    """A river centreline whose banks and levees miss the clearings, else the last draw."""
    v = rng.uniform(-35.0, 35.0)
    for _ in range(OFFSET_TRIES):
        if kit.segment_clear(f.w(-BUILD_HALF_M, v), f.w(BUILD_HALF_M, v), margin=band_half):
            return v
        v = rng.uniform(-35.0, 35.0)
    return v


def _levees(kit: Kit, rng: random.Random, f: Frame, v_r: float, half_w: float) -> int:
    """Earth embankments along both banks, cut wherever they would enter a clearing."""
    zones = _local_zones(kit, f)
    height = rng.uniform(1.2, 2.0)
    n = 0
    for side in (-1.0, 1.0):
        v = v_r + side * (half_w + LEVEE_GAP_M)
        holes = [h for h in (chord_hole(v, zu, zv, kit.clear_radius + LEVEE_HALF_W) for zu, zv in zones) if h]
        parts = [(((a + b) / 2.0, v, height / 2.0 - 0.05), ((b - a) / 2.0, LEVEE_HALF_W, height / 2.0 + 0.05))
                 for a, b in subtract_intervals(-BUILD_HALF_M, BUILD_HALF_M, holes, min_len=2.0)]
        if kit.boxes(parts, LEVEE_RGBA, yaw=f.yaw, bucket="terrain") is not None:
            n += 1
    return n


def _bridge(kit: Kit, rng: random.Random, f: Frame, v_r: float, half_w: float) -> Optional[float]:
    """A road bridge across the river on piers; returns its station along the river, or None."""
    half_len = half_w + LEVEE_GAP_M + LEVEE_HALF_W + 3.0
    for _ in range(OFFSET_TRIES):
        u = rng.uniform(-40.0, 40.0)
        if kit.segment_clear(f.w(u, v_r - half_len), f.w(u, v_r + half_len), margin=4.0):
            break
    else:
        return None
    deck = rng.uniform(4.0, 6.0)
    kit.box(f.w3(u, v_r, deck - 0.4), (2.5, half_len, 0.4), CONCRETE_RGBA, yaw=f.yaw)
    for rail in (-2.4, 2.4):
        kit.box(f.w3(u + rail, v_r, deck + 0.5), (0.06, half_len, 0.5), CONCRETE_RGBA, yaw=f.yaw)
    for end in (-1.0, 1.0):  # abutments carry the deck onto each bank
        kit.box(f.w3(u, v_r + end * (half_len - 1.5), (deck - 0.8) / 2.0), (2.5, 1.5, (deck - 0.8) / 2.0),
                CONCRETE_RGBA, yaw=f.yaw)
    spacing = rng.uniform(8.0, 10.0)
    k = math.floor(half_w / spacing)
    for i in range(-k, k + 1):
        for side in (-1.6, 1.6):
            kit.cylinder(*f.w(u + side, v_r + i * spacing), 0.0, 0.55, deck - 0.8, CONCRETE_RGBA)
    return u


def _river_wire(kit: Kit, rng: random.Random, f: Frame, v_r: float, half_w: float,
                avoid_u: Optional[float]) -> int:
    """An LT crossing strung bank to bank over the water: the classic river-survey hazard."""
    reach = half_w + LEVEE_GAP_M + LEVEE_HALF_W + 3.0
    for _ in range(10):
        u = rng.uniform(-40.0, 40.0)
        if avoid_u is not None and abs(u - avoid_u) < 15.0:
            continue
        a, b = f.w(u, v_r - reach), f.w(u, v_r + reach)
        if kit.clear(*a, 1.0) and kit.clear(*b, 1.0) and kit.segment_clear(a, b, margin=1.0):
            break
    else:
        return 0
    top = rng.uniform(9.0, 10.5)
    for px, py in (a, b):
        kit.cylinder(px, py, 0.0, 0.12, top, CONCRETE_RGBA)
    for dz in (0.3, 0.8):
        kit.wire(WireSpan((a[0], a[1], top - dz), (b[0], b[1], top - dz), rng.uniform(1.2, 2.0), 0.004, "lt"),
                 WIRE_RGBA, n_segments=10)
    return 2


def _bank_trees(kit: Kit, rng: random.Random, f: Frame, v_r: float, half_w: float, season: str) -> int:
    """Mango, peepal and the odd palm along both banks, outside the levees; leaf-off under snow."""
    n = 0
    for _ in range(rng.randint(20, 32)):
        side = rng.choice((-1.0, 1.0))
        u = rng.uniform(-55.0, 55.0)
        v = v_r + side * (half_w + LEVEE_GAP_M + LEVEE_HALF_W + rng.uniform(2.5, 16.0))
        x, y = f.w(u, v)
        crown_r = rng.uniform(2.0, 3.2)
        if not kit.clear(x, y, crown_r):
            continue
        if season == "snow":
            kit.tree(x, y, rng.uniform(7.0, 12.0), crown_r, TRUNK_RGBA, None, crown="none")
        elif rng.random() < 0.2:
            kit.tree(x, y, rng.uniform(8.0, 13.0), crown_r, TRUNK_RGBA, PALM_RGBA, crown="palm")
        else:
            kit.tree(x, y, rng.uniform(7.0, 12.0), crown_r, TRUNK_RGBA, LEAF_RGBA, crown="round")
        n += 1
    return n


def _build_river(kit: Kit, rng: random.Random, f: Frame, season: str) -> Dict[str, object]:
    (w_lo, w_hi), water_rgba = RIVER_BY_SEASON[season]
    half_w = rng.uniform(w_lo, w_hi)
    v_r = _pick_river_axis(kit, rng, f, half_w + LEVEE_GAP_M + LEVEE_HALF_W)
    _water(kit, f, -BUILD_HALF_M, BUILD_HALF_M, v_r - half_w, v_r + half_w, water_rgba)
    if season in ("summer", "winter"):  # low water exposes sandbars mid-channel
        for _ in range(rng.randint(1, 2)):
            u, length = rng.uniform(-40.0, 40.0), rng.uniform(10.0, 25.0)
            v = v_r + rng.uniform(-0.3, 0.3) * half_w
            _water(kit, f, u - length / 2.0, u + length / 2.0, v - 1.5, v + 1.5, SAND_RGBA, z=0.03)
    levees = _levees(kit, rng, f, v_r, half_w)
    bridge_u = _bridge(kit, rng, f, v_r, half_w) if rng.random() < 0.7 else None
    wires = _river_wire(kit, rng, f, v_r, half_w, bridge_u) if rng.random() < 0.4 else 0
    boats = 0
    for _ in range(rng.randint(2, 5)):
        u = rng.uniform(-50.0, 50.0)
        if bridge_u is not None and abs(u - bridge_u) < 8.0:
            continue
        boats += _boat(kit, rng, f, u, v_r + rng.uniform(-0.6, 0.6) * half_w, rng.uniform(-0.3, 0.3))
    trees = _bank_trees(kit, rng, f, v_r, half_w, season)
    return {"river_half_width_m": round(half_w, 2), "levees": levees, "bridge": bridge_u is not None,
            "crossing_wires": wires, "boats": boats, "trees": trees}


# -- coast mode -------------------------------------------------------------------------
def _coast_frame(kit: Kit, rng: random.Random) -> Tuple[Frame, float]:
    """Turn the map so the sea (local +v) lies on the side away from both clearings; returns frame, shoreline."""
    best: Optional[Tuple[Frame, float]] = None
    for _ in range(YAW_TRIES):
        f = Frame(rng.uniform(0.0, 2.0 * math.pi))
        top = max((f.local(x, y)[1] for x, y in kit.safe_zones), default=-10.0)
        if best is None or top < best[1]:
            best = (f, top)
    f, top = best
    return f, top + rng.uniform(6.0, 14.0)


def _palms_and_casuarina(kit: Kit, rng: random.Random, f: Frame, back: float) -> int:
    """A coconut belt behind the beach and, often, a casuarina windbreak along its back edge."""
    n = 0
    for _ in range(rng.randint(18, 28)):
        u, v = rng.uniform(-55.0, 55.0), back - rng.uniform(0.0, 16.0)
        x, y = f.w(u, v)
        if kit.clear(x, y, 2.5):
            kit.tree(x, y, rng.uniform(8.0, 14.0), rng.uniform(2.2, 3.0), TRUNK_RGBA, PALM_RGBA, crown="palm")
            n += 1
    if rng.random() < 0.5:
        start = rng.uniform(-50.0, 0.0)
        u = start
        while u < start + 50.0:
            x, y = f.w(u, back + rng.uniform(0.5, 2.5))
            if kit.clear(x, y, 1.4):
                kit.tree(x, y, rng.uniform(10.0, 14.0), rng.uniform(1.0, 1.4), TRUNK_RGBA, CASUARINA_RGBA,
                         crown="cone")
                n += 1
            u += rng.uniform(3.0, 5.0)
    return n


def _jetty(kit: Kit, rng: random.Random, f: Frame, shore: float) -> bool:
    """A timber jetty from the beach out to sea on pairs of piles."""
    length = rng.uniform(20.0, 35.0)
    for _ in range(OFFSET_TRIES):
        u = rng.uniform(-40.0, 40.0)
        if kit.segment_clear(f.w(u, shore - 4.0), f.w(u, shore + length), margin=2.0):
            break
    else:
        return False
    deck = rng.uniform(2.5, 3.5)
    v0, v1 = shore - 4.0, shore + length
    kit.box(f.w3(u, (v0 + v1) / 2.0, deck - 0.15), (1.5, (v1 - v0) / 2.0, 0.15), TIMBER_RGBA, yaw=f.yaw)
    v = v0 + 2.0
    while v < v1:
        for side in (-1.3, 1.3):
            kit.cylinder(*f.w(u + side, v), 0.0, 0.15, deck - 0.3, TIMBER_RGBA)
        v += 5.0
    return True


def _lighthouse(kit: Kit, rng: random.Random, f: Frame, back: float) -> bool:
    u, v = rng.uniform(-45.0, 45.0), back - rng.uniform(4.0, 10.0)
    x, y = f.w(u, v)
    if not kit.clear(x, y, 3.0):
        return False
    h = rng.uniform(18.0, 24.0)
    kit.cylinder(x, y, 0.0, 1.8, 0.6 * h, WHITE_RGBA)
    kit.cylinder(x, y, 0.6 * h, 1.5, 0.35 * h, RED_RGBA)
    kit.cylinder(x, y, 0.95 * h, 1.1, 0.05 * h + 1.5, WHITE_RGBA)
    return True


def _build_coast(kit: Kit, rng: random.Random, f: Frame, shore: float, season: str) -> Dict[str, object]:
    beach_w = rng.uniform(10.0, 20.0)
    back = shore - beach_w
    _water(kit, f, -BUILD_HALF_M - 40.0, BUILD_HALF_M + 40.0, shore, shore + 120.0, SEA_RGBA_BY_SEASON[season])
    _water(kit, f, -BUILD_HALF_M - 40.0, BUILD_HALF_M + 40.0, shore - 0.8, shore + 0.8, SURF_RGBA, z=0.03)
    _water(kit, f, -BUILD_HALF_M - 40.0, BUILD_HALF_M + 40.0, back, shore - 0.8, SAND_RGBA, z=0.015)
    trees = _palms_and_casuarina(kit, rng, f, back)
    boats = 0
    for _ in range(rng.randint(3, 6)):  # hauled up on the sand, bow to the sea
        boats += _boat(kit, rng, f, rng.uniform(-50.0, 50.0), rng.uniform(back + 2.0, shore - 2.0), math.pi / 2.0)
    for _ in range(rng.randint(1, 3)):  # moored offshore
        boats += _boat(kit, rng, f, rng.uniform(-50.0, 50.0), shore + rng.uniform(12.0, 40.0), rng.uniform(0, math.pi))
    jetty = _jetty(kit, rng, f, shore) if rng.random() < 0.6 else False
    lighthouse = _lighthouse(kit, rng, f, back) if rng.random() < 0.5 else False
    huts = 0
    for _ in range(rng.randint(4, 8)):
        x, y = f.w(rng.uniform(-50.0, 50.0), back - rng.uniform(12.0, 26.0))
        if kit.clear(x, y, 3.5):
            kit.hut(x, y, rng.uniform(3.0, 4.5), rng.uniform(3.0, 4.0), rng.uniform(2.4, 3.0), WALL_RGBA,
                    THATCH_RGBA, yaw=f.yaw + rng.uniform(-0.2, 0.2))
            huts += 1
    return {"beach_width_m": round(beach_w, 2), "trees": trees, "boats": boats, "jetty": jetty,
            "lighthouse": lighthouse, "huts": huts}


def build_river_coast_map(cli: int, seed: int, safe_zones, safe_zone_radius: float) -> dict:
    """Build the type 11 river or coast for a seed; returns the asset dict the map dispatcher tags."""
    rng = random.Random((int(seed) + RIVER_COAST_SEED_OFFSET) & 0xFFFFFFFF)
    season = scenario_for_seed(int(seed), RIVER_COAST_CHALLENGE_TYPE).season
    mode = "coast" if season != "snow" and rng.random() < COAST_PROBABILITY else "river"

    kit = Kit(cli, safe_zones, safe_zone_radius)
    kit.ground(GROUND_RGBA_BY_SEASON[season])
    if mode == "river":
        f = Frame(rng.uniform(0.0, math.pi))
        facts = _build_river(kit, rng, f, season)
    else:
        f, shore = _coast_frame(kit, rng)
        facts = _build_coast(kit, rng, f, shore, season)
    return kit.summary(biome="river_coast", mode=mode, season=season, yaw=round(f.yaw, 3), **facts)


__all__ = ["RIVER_COAST_CHALLENGE_TYPE", "build_river_coast_map"]
