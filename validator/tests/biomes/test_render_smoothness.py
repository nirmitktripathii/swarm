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

"""What makes a video clip smooth: even frame spacing and a chase camera that does not shake with the airframe."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pybullet as p
import pytest

import validator.scripts.generate_video as generate_video
from swarm.constants import SIM_DT

SIDEWAYS_TREMOR_M = 0.05
FRAME_DT = 1.0 / generate_video.DEFAULT_FPS


def _eye_path(monkeypatch, position_smoothing: float, frames: int = 80):
    """Camera eye positions for a drone cruising along +x with a sideways tremor of alternating sign."""
    monkeypatch.setattr(generate_video, "CHASE_POSITION_SMOOTHING", position_smoothing)
    monkeypatch.setattr(generate_video, "_render_rgb", lambda *args, **kwargs: np.zeros((2, 2, 3), np.uint8))
    cli = p.connect(p.DIRECT)
    try:
        cam = generate_video.ChaseCamera(cli, 64, 48)
        eyes = []
        real_view = cam._view
        cam._view = lambda eye, target, up=(0, 0, 1): (eyes.append(np.array(eye)), real_view(eye, target, up))[1]
        for i in range(frames):
            pos = np.array([5.0 * FRAME_DT * i, SIDEWAYS_TREMOR_M * (-1) ** i, 2.0])
            cam.capture(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.eye(3), FRAME_DT)
        return np.array(eyes)
    finally:
        p.disconnect(cli)


def _second_difference_rms(path: np.ndarray) -> float:
    return float(np.sqrt((np.diff(path, n=2, axis=0) ** 2).sum(axis=1).mean()))


def test_position_smoothing_takes_the_tremor_out_of_the_camera_path(monkeypatch):
    raw = _eye_path(monkeypatch, 0.0)
    smooth = _eye_path(monkeypatch, 0.5)

    assert _second_difference_rms(smooth) < 0.5 * _second_difference_rms(raw)


def test_position_smoothing_lags_steady_flight_by_well_under_a_metre(monkeypatch):
    raw = _eye_path(monkeypatch, 0.0)
    smooth = _eye_path(monkeypatch, 0.5)

    lag_x = raw[-1, 0] - smooth[-1, 0]
    assert 0.0 <= lag_x < 0.5, f"camera trails the drone by {lag_x:.2f} m"


def test_the_default_smoothing_is_on():
    assert 0.0 < generate_video.CHASE_POSITION_SMOOTHING < 1.0


def test_the_phase_summary_names_every_phase_and_the_frame_count(monkeypatch):
    monkeypatch.setattr(generate_video, "_PHASE_SEC",
                        {"agent": 1.25, "step": 2.0, "raster": 30.5, "haze": 0.0, "encode": 0.4})

    line = generate_video._phase_summary(120, 40.04)

    assert line == "agent=1.2s  step=2.0s  raster=30.5s  haze=0.0s  encode=0.4s  frames=120  wall=40.0s"


@pytest.mark.parametrize("fps", [10, 25, 50])  # whole numbers of 50 Hz steps per frame
def test_frames_land_on_every_nth_step_with_no_float_drift(fps):
    """The loop's own due-test, over a long flight: after the first frame, one constant gap in steps."""
    steps_per_frame = round(1.0 / (fps * SIM_DT))
    frame_dt = 1.0 / fps
    t, next_frame_t, captured = 0.0, 0.0, []
    for step in range(1, 6001):
        t += float(SIM_DT)
        if t + generate_video._FRAME_EPS >= next_frame_t:
            captured.append(step)
            next_frame_t += frame_dt

    steady = captured[1:]  # the first frame is taken at step 1, the second at step steps_per_frame
    gaps = Counter(b - a for a, b in zip(steady, steady[1:]))
    assert gaps == {steps_per_frame: len(steady) - 1}, gaps
