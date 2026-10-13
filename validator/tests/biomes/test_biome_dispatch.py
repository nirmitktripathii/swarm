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

"""The biome types wired into the sim: opt-in registration, map dispatch and tagging, wires, and pad placement."""
from __future__ import annotations

import collections
import math

import pybullet as p
import pytest

pytest.importorskip("swarm_worlds")  # the pads and the stock maps load their meshes and textures from it

from swarm.constants import CHALLENGE_TYPE_DISTRIBUTION  # noqa: E402
from swarm.core.env_builder.platform_placement import build_autopilot_world  # noqa: E402
from swarm.core.env_builder.sar_tagging import BIOME_CHALLENGE_TYPES, build_and_tag_map  # noqa: E402
from swarm.core.tof_wire_model import wires_for  # noqa: E402
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE  # noqa: E402
from swarm.validator.task_gen import task_for_seed_and_type  # noqa: E402

SIM_DT = 0.02
PLACEMENT_SEEDS = range(1, 6)
ENVIRONMENTS = {8: "farmland", 9: "powerline", 10: "terraced_hills", 11: "river_coast"}


def _tags(tagger):
    return collections.Counter(tagger.body_tags.values())


def test_biome_types_are_registered_but_opt_in():
    assert BIOME_CHALLENGE_TYPES == frozenset(ENVIRONMENTS)
    for challenge_type, environment in ENVIRONMENTS.items():
        assert CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE[challenge_type] == environment
        assert challenge_type not in CHALLENGE_TYPE_DISTRIBUTION, "biomes must never be drawn by the stock sampler"


def test_an_unknown_challenge_type_is_rejected(cli):
    with pytest.raises(ValueError):
        build_and_tag_map(cli, 1, max(ENVIRONMENTS) + 1)


@pytest.mark.parametrize("challenge_type", sorted(ENVIRONMENTS))
def test_tasks_generate_for_every_biome(challenge_type):
    task = task_for_seed_and_type(SIM_DT, seed=7, challenge_type=challenge_type)
    assert task.challenge_type == challenge_type
    assert all(math.isfinite(v) for v in (*task.start, *task.goal))


def test_powerline_build_tags_its_wires_and_a_rebuild_clears_them(cli):
    task = task_for_seed_and_type(SIM_DT, seed=3, challenge_type=9)
    tagger = build_and_tag_map(cli, task.map_seed, 9, start=task.start, goal=task.goal, safe_zone_radius=0.0)
    tags = _tags(tagger)
    assert wires_for(cli), "a powerline map registers its spans for the ToF model"
    assert tags["OBSTACLE_BEAM"] == len(wires_for(cli))
    assert tags["SUPPORT_TERRAIN"] >= 1

    p.resetSimulation(physicsClientId=cli)
    task = task_for_seed_and_type(SIM_DT, seed=3, challenge_type=10)
    tagger = build_and_tag_map(cli, task.map_seed, 10, start=task.start, goal=task.goal, safe_zone_radius=0.0)
    assert wires_for(cli) == (), "a map without wires must not inherit the last map's spans"
    assert _tags(tagger)["SUPPORT_TERRAIN"] > 1, "terrace treads are terrain"


def _inside_a_body(cli, xyz, ignore, radius=0.1):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=cli)
    probe = p.createMultiBody(0, col, -1, list(xyz), physicsClientId=cli)
    try:
        return [uid for uid in range(p.getNumBodies(physicsClientId=cli))
                if uid not in ignore and uid != probe and p.getClosestPoints(probe, uid, 0.0, physicsClientId=cli)]
    finally:
        p.removeBody(probe, physicsClientId=cli)


@pytest.mark.parametrize("challenge_type", sorted(ENVIRONMENTS))
def test_pads_stay_in_their_clearings_with_a_free_start(challenge_type):
    for seed in PLACEMENT_SEEDS:
        task = task_for_seed_and_type(SIM_DT, seed=seed, challenge_type=challenge_type)
        cli = p.connect(p.DIRECT)
        try:
            base = p.getNumBodies(physicsClientId=cli)
            tagger = build_and_tag_map(cli, task.map_seed, challenge_type, start=task.start, goal=task.goal,
                                       safe_zone_radius=0.0)
            end_uids, start_uids, _, _, adj_start, adj_goal = build_autopilot_world(
                tagger, cli, task.map_seed, task.start, task.goal, challenge_type, False, base)
            start, goal = adj_start or task.start, adj_goal or task.goal
            assert start_uids and end_uids, f"type {challenge_type} seed {seed}: missing pads"
            # Clearings are carved at the requested xy; a pad moved off it would lose that guarantee.
            assert start[:2] == pytest.approx(task.start[:2], abs=1e-6), (challenge_type, seed, start, task.start)
            assert goal[:2] == pytest.approx(task.goal[:2], abs=1e-6), (challenge_type, seed, goal, task.goal)
            hits = _inside_a_body(cli, start, ignore=set(start_uids))
            assert not hits, f"type {challenge_type} seed {seed}: drone spawns inside bodies {hits}"
        finally:
            p.disconnect(cli)
