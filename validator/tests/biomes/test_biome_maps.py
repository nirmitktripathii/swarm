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

"""The biome maps: kit helpers, wires that collide, determinism, clearings, and level terrace pads."""
from __future__ import annotations

import math
import random

import pybullet as p
import pytest

from swarm.core.maps.biome_kit import (
    CLEARING_MIN_M,
    Frame,
    Kit,
    WireSpan,
    catenary_points,
    chord_hole,
    segment_clear,
    subtract_intervals,
)
from swarm.core.maps.terraced_hills.builder import cut_square, plan_terraces

from .conftest import BIOMES, ZONES, biome_builder

SEEDS = range(1, 7)


# -- pure helpers ---------------------------------------------------------------------------
def test_subtract_intervals_removes_holes_and_short_pieces():
    assert subtract_intervals(0.0, 10.0, [(2.0, 3.0), (5.0, 9.5)]) == [(0.0, 2.0), (3.0, 5.0), (9.5, 10.0)]
    assert subtract_intervals(0.0, 10.0, [(2.0, 3.0), (5.0, 9.5)], min_len=1.0) == [(0.0, 2.0), (3.0, 5.0)]
    assert subtract_intervals(0.0, 10.0, [(-5.0, 20.0)]) == []


def test_chord_hole_is_the_stretch_inside_the_circle():
    lo, hi = chord_hole(line_coord=3.0, centre_along=10.0, centre_across=0.0, radius=5.0)
    assert (lo, hi) == pytest.approx((6.0, 14.0))
    assert chord_hole(line_coord=5.0, centre_along=0.0, centre_across=0.0, radius=5.0) is None


def test_catenary_hits_both_ends_and_sags_at_mid_span():
    pts = catenary_points((0.0, 0.0, 10.0), (20.0, 0.0, 10.0), sag_m=0.8, n_segments=8)
    assert pts[0] == (0.0, 0.0, 10.0) and pts[-1] == (20.0, 0.0, 10.0)
    assert pts[4][2] == pytest.approx(9.2)
    with pytest.raises(ValueError):
        catenary_points((0, 0, 0), (1, 0, 0), 0.1, 0)


def test_segment_clear_rejects_a_line_through_a_clearing():
    assert not segment_clear((-20.0, 1.0), (20.0, 1.0), [(0.0, 0.0)], 8.0)
    assert segment_clear((-20.0, 9.0), (20.0, 9.0), [(0.0, 0.0)], 8.0)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.4, math.pi])
def test_frame_local_inverts_world(yaw):
    f = Frame(yaw)
    x, y = f.w(12.5, -3.0)
    assert f.local(x, y) == pytest.approx((12.5, -3.0))


def test_wire_is_one_body_that_collides_along_its_sag(cli):
    kit = Kit(cli, [], 0.0)
    span = WireSpan((0.0, -10.0, 10.0), (0.0, 10.0, 10.0), sag_m=1.0, radius_m=0.005, kind="lt")
    uids = kit.wire(span, (0.1, 0.1, 0.1, 1.0))
    assert len(uids) == 1 and kit.out.wires == [span] and uids[0] in kit.out.beams
    hit = p.rayTest([-5.0, 0.0, 9.0], [5.0, 0.0, 9.0], physicsClientId=cli)[0]
    assert hit[0] == uids[0], "a 5 mm conductor must still stop a drone at its sagged height"
    miss = p.rayTest([-5.0, 0.0, 9.6], [5.0, 0.0, 9.6], physicsClientId=cli)[0]
    assert miss[0] == -1


# -- terrace geometry -------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_terrace_plan_climbs_monotonically(seed):
    plan = plan_terraces(random.Random(seed))
    assert all(b > a for a, b in zip(plan.edges, plan.edges[1:]))
    assert all(b > a for a, b in zip(plan.tops, plan.tops[1:]))
    assert len(plan.edges) == plan.n_treads + 1


def test_cut_square_keeps_the_area_outside_the_notch():
    pieces = cut_square((0.0, 10.0, 0.0, 10.0), 5.0, 5.0, 2.0)
    assert sum((u1 - u0) * (v1 - v0) for u0, u1, v0, v1 in pieces) == pytest.approx(100.0 - 16.0)
    assert cut_square((0.0, 10.0, 0.0, 10.0), 30.0, 30.0, 2.0) == [(0.0, 10.0, 0.0, 10.0)]


# -- built maps -------------------------------------------------------------------------------
def _aabbs(cli, out):
    uids = [u for key in ("trees", "props", "beams", "terrain") for u in out[key]]
    return sorted(tuple(round(c, 4) for c in p.getAABB(u, physicsClientId=cli)[0] + p.getAABB(u, physicsClientId=cli)[1])
                  for u in uids)


@pytest.mark.parametrize("challenge_type", sorted(BIOMES))
def test_same_seed_builds_the_same_map(challenge_type):
    build = biome_builder(challenge_type)
    worlds = []
    for _ in range(2):
        cli = p.connect(p.DIRECT)
        try:
            out = build(cli, 4242, ZONES, 0.0)
            worlds.append((out["info"], _aabbs(cli, out), out["wires"]))
        finally:
            p.disconnect(cli)
    assert worlds[0] == worlds[1]


@pytest.mark.parametrize("challenge_type", sorted(BIOMES))
def test_start_and_goal_clearings_stay_open(challenge_type):
    """Nothing a drone can hit stands within 7.5 m of a pad up to 8 m above it, on any seed."""
    build = biome_builder(challenge_type)
    for seed in SEEDS:
        cli = p.connect(p.DIRECT)
        try:
            out = build(cli, seed, ZONES, 0.0)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=7.5, height=8.0, physicsClientId=cli)
            for zx, zy in ZONES:
                probe = p.createMultiBody(0, col, -1, [zx, zy, 4.2], physicsClientId=cli)
                hits = [uid for key in ("trees", "props", "beams") for uid in out[key]
                        if p.getClosestPoints(probe, uid, 0.0, physicsClientId=cli)]
                assert not hits, f"type {challenge_type} seed {seed}: bodies {hits} inside clearing {(zx, zy)}"
                p.removeBody(probe, physicsClientId=cli)
        finally:
            p.disconnect(cli)


@pytest.mark.parametrize("challenge_type", sorted(BIOMES))
def test_no_wire_crosses_a_clearing(challenge_type):
    build = biome_builder(challenge_type)
    for seed in SEEDS:
        cli = p.connect(p.DIRECT)
        try:
            out = build(cli, seed, ZONES, 0.0)
        finally:
            p.disconnect(cli)
        for span in out["wires"]:
            assert segment_clear(span.a[:2], span.b[:2], ZONES, CLEARING_MIN_M), (challenge_type, seed, span)
            assert span.radius_m < 0.02, "real conductors are millimetres thick"


def test_powerline_maps_always_carry_wires():
    build = biome_builder(9)
    for seed in SEEDS:
        cli = p.connect(p.DIRECT)
        try:
            out = build(cli, seed, ZONES, 0.0)
        finally:
            p.disconnect(cli)
        assert out["wires"], f"seed {seed} built a powerline map with no wires"
        assert out["info"]["n_wires"] == len(out["wires"])


@pytest.mark.parametrize("seed", SEEDS)
def test_terrace_pads_are_level(cli, seed):
    biome_builder(10)(cli, seed, ZONES, 0.0)
    for zx, zy in ZONES:
        zs = [p.rayTest([zx + dx, zy + dy, 80.0], [zx + dx, zy + dy, -1.0], physicsClientId=cli)[0][3][2]
              for dx in (-1.2, 0.0, 1.2) for dy in (-1.2, 0.0, 1.2)]
        assert max(zs) - min(zs) < 1e-3, f"seed {seed}: pad at {(zx, zy)} not level: {zs}"
