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

"""The opt-in flight trace: what a row and the header hold, and that a row is plain JSON."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import validator.scripts.generate_video as generate_video


def _fake_env():
    state = np.zeros(20)
    state[0:3] = [1.0, 2.0, 3.0]
    state[7:10] = [0.01, -0.02, 0.0]
    state[10:13] = [0.5, 0.0, -0.25]
    return SimpleNamespace(
        _getDroneStateVector=lambda index: state,
        _current_platform_pos=np.array([4.0, 5.0, 0.6], dtype=np.float32),
        _platform_velocity=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        _platform_hit=False, _success=False, _failure_reason="",
        _moving=True, _movement_pattern="figure8", _platform_speed=0.9, _platform_radius=3.0,
        _platform_delay=2.0, _platform_transition_time=3.0, _platform_phase=1.5,
        _platform_orbit_center=np.array([4.0, 5.0, 0.6], dtype=np.float32),
    )


def test_a_row_carries_drone_and_pad_state_and_is_plain_json():
    row = generate_video._trace_row(_fake_env(), 12.3456)

    assert json.loads(json.dumps(row)) == row
    assert row["t"] == 12.346
    assert row["pos"] == [1.0, 2.0, 3.0]
    assert row["vel"] == [0.5, 0.0, -0.25]
    assert row["pad"] == [4.0, 5.0, 0.6]
    assert (row["hit"], row["success"], row["fail"]) == (False, False, "")


def test_the_header_names_the_pad_motion():
    task = SimpleNamespace(horizon=90.0, start=(0.0, 0.0, 1.0), goal=(10.0, 0.0, 0.6))

    header = generate_video._trace_header(_fake_env(), task)

    assert json.loads(json.dumps(header)) == header
    assert header["moving"] is True
    assert header["pattern"] == "figure8"
    assert header["orbit_center"] == [4.0, 5.0, 0.6]
    assert header["horizon"] == 90.0
