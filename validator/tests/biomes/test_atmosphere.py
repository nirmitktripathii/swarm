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

"""Distance fog and the sky gradient applied to a finished frame from its depth buffer."""
from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from swarm.core.atmosphere import (
    Atmosphere,
    apply_atmosphere,
    atmosphere_for,
    depth_metres,
    projection_planes,
)
from swarm.core.scenario import sample_scenario

NEAR, FAR = 0.1, 100.0
FOG = (0.8, 0.8, 0.8)
HORIZON, ZENITH = (0.9, 0.7, 0.5), (0.1, 0.2, 0.6)


def _z_at(distance_m: float) -> float:
    """Depth-buffer value of a surface `distance_m` in front of the camera."""
    return (FAR - FAR * NEAR / distance_m) / (FAR - NEAR)


def _frame(value: int, h: int = 6, w: int = 4) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_projection_planes_recover_near_and_far_from_pybullet_matrices():
    proj = p.computeProjectionMatrixFOV(60.0, 1.6, NEAR, FAR)

    near, far = projection_planes(proj)

    assert near == pytest.approx(NEAR, rel=1e-4)
    assert far == pytest.approx(FAR, rel=1e-4)


def test_projection_planes_reject_a_matrix_of_the_wrong_size():
    with pytest.raises(ValueError):
        projection_planes([1.0] * 9)


@pytest.mark.parametrize("distance", [0.5, 5.0, 60.0])
def test_depth_metres_inverts_the_depth_buffer(distance):
    assert depth_metres(np.array([_z_at(distance)]), NEAR, FAR)[0] == pytest.approx(distance, rel=1e-3)


def test_a_near_surface_is_barely_touched_and_a_far_one_goes_to_the_fog_colour():
    atm = Atmosphere(haze_m=50.0, fog_rgb=FOG, sky=None)
    z = np.stack([np.full(4, _z_at(1.0)), np.full(4, _z_at(90.0))] * 3).astype(np.float32)  # rows: near, far

    out = apply_atmosphere(_frame(0), z, NEAR, FAR, atm)

    near_row, far_row = out[0], out[1]
    assert near_row.max() < 0.05 * 255, "1 m of 50 m haze keeps ~98% of the surface"
    assert far_row.min() > 0.6 * 255, "90 m of 50 m haze is mostly fog"


def test_clear_air_leaves_the_frame_alone():
    atm = Atmosphere(haze_m=6000.0, fog_rgb=FOG, sky=None)
    z = np.full((6, 4), _z_at(30.0), dtype=np.float32)
    frame = _frame(120)

    out = apply_atmosphere(frame, z, NEAR, FAR, atm)

    assert np.abs(out.astype(int) - frame.astype(int)).max() <= 3


def test_the_sky_gradient_runs_from_zenith_at_the_top_to_horizon_at_the_lowest_sky_row():
    atm = Atmosphere(haze_m=50.0, fog_rgb=FOG, sky=(HORIZON, ZENITH))
    z = np.ones((6, 4), dtype=np.float32)
    z[4:] = _z_at(10.0)  # rows 4-5 are ground, rows 0-3 are sky

    out = apply_atmosphere(_frame(0), z, NEAR, FAR, atm)

    np.testing.assert_allclose(out[0, 0], np.rint(np.array(ZENITH) * 255), atol=1)
    np.testing.assert_allclose(out[3, 0], np.rint(np.array(HORIZON) * 255), atol=1)
    assert out[1, 0, 0] > out[0, 0, 0], "red climbs from zenith toward the horizon"


def test_without_a_sky_the_renderers_own_sky_pixels_are_kept():
    atm = Atmosphere(haze_m=50.0, fog_rgb=FOG, sky=None)
    z = np.ones((6, 4), dtype=np.float32)
    frame = _frame(77)

    out = apply_atmosphere(frame, z, NEAR, FAR, atm)

    np.testing.assert_array_equal(out, frame)


def test_the_input_frame_is_not_modified_and_the_output_is_a_new_uint8_array():
    atm = Atmosphere(haze_m=20.0, fog_rgb=FOG, sky=(HORIZON, ZENITH))
    frame = _frame(10)
    snapshot = frame.copy()

    out = apply_atmosphere(frame, np.full((6, 4), _z_at(40.0), dtype=np.float32), NEAR, FAR, atm)

    np.testing.assert_array_equal(frame, snapshot)
    assert out.dtype == np.uint8 and out.shape == frame.shape and out is not frame


def test_an_rgba_frame_comes_back_as_rgb():
    atm = Atmosphere(haze_m=20.0, fog_rgb=FOG, sky=None)
    rgba = np.zeros((6, 4, 4), dtype=np.uint8)

    out = apply_atmosphere(rgba, np.full((6, 4), _z_at(10.0), dtype=np.float32), NEAR, FAR, atm)

    assert out.shape == (6, 4, 3)


@pytest.mark.parametrize("haze", [0.0, -5.0, float("nan")])
def test_a_non_positive_haze_distance_is_rejected(haze):
    with pytest.raises(ValueError):
        Atmosphere(haze_m=haze, fog_rgb=FOG, sky=None)


def test_fog_is_far_thicker_than_clear_air_for_a_scenario_spec():
    clear = atmosphere_for(sample_scenario(7, 8, season="summer", time_of_day="day", weather="clear"))
    fog = atmosphere_for(sample_scenario(7, 8, season="summer", time_of_day="day", weather="fog"))

    assert fog.haze_m * 10 < clear.haze_m
    assert fog.sky is not None and clear.sky is None


def test_night_fades_toward_a_dark_colour_not_a_pale_one():
    night = atmosphere_for(sample_scenario(7, 8, time_of_day="night", weather="clear"))
    day = atmosphere_for(sample_scenario(7, 8, time_of_day="day", weather="haze"))

    assert max(night.fog_rgb) < 0.3 < min(day.fog_rgb)
