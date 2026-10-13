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

"""Shared fixtures for the biome map tests: a headless physics client and the four builders."""
from __future__ import annotations

import importlib

import pybullet as p
import pytest

# challenge type -> (module, builder function)
BIOMES = {
    8: ("swarm.core.maps.farmland", "build_farmland_map"),
    9: ("swarm.core.maps.powerline", "build_powerline_map"),
    10: ("swarm.core.maps.terraced_hills", "build_terraced_hills_map"),
    11: ("swarm.core.maps.river_coast", "build_river_coast_map"),
}

# Start and goal clearings used by the builder-level tests: one at the origin, one off-axis.
ZONES = [(0.0, 0.0), (30.0, 10.0)]


def biome_builder(challenge_type: int):
    module, func = BIOMES[challenge_type]
    return getattr(importlib.import_module(module), func)


@pytest.fixture
def cli():
    client = p.connect(p.DIRECT)
    try:
        yield client
    finally:
        p.disconnect(client)
