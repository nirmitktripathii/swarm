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

"""The scenario's light, sky and haze as renderer arguments, and the opt-in that feeds them to the video cameras."""
from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

import validator.scripts.generate_video as generate_video
from swarm.core.scenario import render_kwargs, sample_scenario

SEED, CHALLENGE_TYPE = 7, 8
FIXED_LIGHT = [0.4, 0.4, 1.0]
# the keyword arguments stock pybullet's getCameraImage takes, as used by the video cameras
STOCK_KWARGS = {"width", "height", "viewMatrix", "projectionMatrix", "renderer", "shadow",
                "physicsClientId", "lightDirection", "lightColor", "lightAmbientCoeff",
                "lightDiffuseCoeff"}
PROJECTION = p.computeProjectionMatrixFOV(60.0, 1.6, 0.1, 100.0)


def test_a_clear_day_sets_the_light_and_haze_but_no_sky():
    spec = sample_scenario(SEED, CHALLENGE_TYPE, season="summer", time_of_day="day", weather="clear")

    kwargs = render_kwargs(spec)

    assert kwargs["lightDirection"] == list(spec.sun.direction)
    assert kwargs["lightColor"] == list(spec.sun.color)
    assert kwargs["lightAmbientCoeff"] == spec.sun.ambient
    assert kwargs["lightDiffuseCoeff"] == spec.sun.diffuse
    assert kwargs["hazeDistance"] == spec.haze_m
    assert "skyHorizonColor" not in kwargs and "skyZenithColor" not in kwargs, "clear day keeps the renderer's sky"


@pytest.mark.parametrize("weather", ["haze", "fog", "rain", "dust", "snowfall"])
def test_weather_with_a_sky_passes_its_colours(weather):
    spec = sample_scenario(SEED, CHALLENGE_TYPE, time_of_day="day", weather=weather)

    kwargs = render_kwargs(spec)

    horizon, zenith = spec.sky()
    assert kwargs["skyHorizonColor"] == list(horizon)
    assert kwargs["skyZenithColor"] == list(zenith)


def test_night_carries_the_moons_dark_sky_whatever_the_weather():
    night = render_kwargs(sample_scenario(SEED, CHALLENGE_TYPE, time_of_day="night", weather="clear"))
    day = render_kwargs(sample_scenario(SEED, CHALLENGE_TYPE, time_of_day="day", weather="clear"))

    assert "skyZenithColor" in night
    assert max(night["skyHorizonColor"]) < 0.3, "a night sky is dark"
    assert night["lightAmbientCoeff"] < day["lightAmbientCoeff"]


def test_fog_is_far_thicker_than_clear_air():
    clear = render_kwargs(sample_scenario(SEED, CHALLENGE_TYPE, weather="clear"))
    fog = render_kwargs(sample_scenario(SEED, CHALLENGE_TYPE, weather="fog"))

    assert fog["hazeDistance"] * 10 < clear["hazeDistance"]


def _fake_camera(captured, depth=1.0):
    """A stock-pybullet stand-in: records its arguments and fills the depth buffer with `depth`."""
    def get_camera_image(**kwargs):
        captured.update(kwargs)
        h, w = kwargs["height"], kwargs["width"]
        return w, h, np.zeros((h, w, 4), dtype=np.uint8), np.full((h, w), depth, dtype=np.float32), None
    return get_camera_image


@pytest.fixture(autouse=True)
def _clean_look(monkeypatch):
    monkeypatch.delenv(generate_video.ENV_SCENARIO_RENDER, raising=False)
    monkeypatch.setattr(generate_video, "_ATMOSPHERE", None)
    generate_video._RENDER_LOOK.clear()
    yield
    generate_video._RENDER_LOOK.clear()


def test_without_the_switch_the_cameras_keep_the_fixed_light(monkeypatch):
    captured = {}
    monkeypatch.setattr(p, "getCameraImage", _fake_camera(captured))

    assert generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE) is None
    frame = generate_video._render_rgb(0, None, PROJECTION, 8, 6)

    assert captured["lightDirection"] == FIXED_LIGHT
    assert set(captured) <= STOCK_KWARGS and "lightColor" not in captured
    assert not frame.any(), "the frame is untouched"


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_with_the_switch_the_cameras_take_the_scenarios_light(monkeypatch, value):
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, value)
    monkeypatch.setenv("SWARM_SCENARIO_WEATHER", "fog")
    monkeypatch.setenv("SWARM_SCENARIO_TIME", "dusk")
    captured = {}
    monkeypatch.setattr(p, "getCameraImage", _fake_camera(captured))

    spec = generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
    generate_video._render_rgb(0, None, PROJECTION, 8, 6)

    assert (spec.weather, spec.time_of_day) == ("fog", "dusk")
    assert captured["lightDirection"] == list(spec.sun.direction) != FIXED_LIGHT
    assert captured["lightColor"] == list(spec.sun.color)


def test_renderer_only_arguments_never_reach_stock_pybullet(monkeypatch):
    """Kaggle's pybullet raised TypeError on hazeDistance; haze and sky are applied in software instead."""
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, "1")
    monkeypatch.setenv("SWARM_SCENARIO_WEATHER", "fog")
    captured = {}
    monkeypatch.setattr(p, "getCameraImage", _fake_camera(captured))

    generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
    generate_video._render_rgb(0, None, PROJECTION, 8, 6)

    assert set(captured) <= STOCK_KWARGS, set(captured) - STOCK_KWARGS


def test_fog_paints_the_sky_gradient_over_empty_pixels(monkeypatch):
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, "1")
    monkeypatch.setenv("SWARM_SCENARIO_WEATHER", "fog")
    monkeypatch.setenv("SWARM_SCENARIO_TIME", "day")
    monkeypatch.setattr(p, "getCameraImage", _fake_camera({}, depth=1.0))

    spec = generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
    frame = generate_video._render_rgb(0, None, PROJECTION, 8, 6)

    horizon, zenith = spec.sky()
    np.testing.assert_allclose(frame[0, 0], np.rint(np.array(zenith) * 255), atol=1)
    np.testing.assert_allclose(frame[-1, 0], np.rint(np.array(horizon) * 255), atol=1)


def test_fog_hides_a_distant_surface_but_a_clear_day_does_not(monkeypatch):
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, "1")
    monkeypatch.setenv("SWARM_SCENARIO_TIME", "day")
    z_at_60m = (100.0 - 100.0 * 0.1 / 60.0) / (100.0 - 0.1)  # depth-buffer value of a surface 60 m out
    monkeypatch.setattr(p, "getCameraImage", _fake_camera({}, depth=z_at_60m))

    def surface_brightness(weather):
        monkeypatch.setenv("SWARM_SCENARIO_WEATHER", weather)
        generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
        return generate_video._render_rgb(0, None, PROJECTION, 8, 6).astype(float).mean()

    clear, fog = surface_brightness("clear"), surface_brightness("fog")

    assert clear < 5.0, "a black surface stays black through clear air"
    assert fog > 50.0, "fog fades it toward the fog colour"


def test_the_look_is_reset_for_the_next_flight(monkeypatch):
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, "1")
    generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
    assert generate_video._RENDER_LOOK and generate_video._ATMOSPHERE is not None

    monkeypatch.delenv(generate_video.ENV_SCENARIO_RENDER)
    generate_video._apply_scenario_look(SEED + 1, CHALLENGE_TYPE)

    assert generate_video._RENDER_LOOK == {} and generate_video._ATMOSPHERE is None


@pytest.mark.parametrize("challenge_type", [8, 9, 10, 11])
def test_the_video_cli_accepts_the_biome_maps(challenge_type):
    assert challenge_type in generate_video.TYPE_LABELS
    args = generate_video._build_parser().parse_args(
        ["--model", "m.zip", "--seed", "1", "--type", str(challenge_type)])
    assert args.type == challenge_type
