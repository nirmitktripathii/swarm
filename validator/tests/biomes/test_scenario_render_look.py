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


def _fake_camera(captured):
    def get_camera_image(**kwargs):
        captured.update(kwargs)
        h, w = kwargs["height"], kwargs["width"]
        return w, h, np.zeros((h, w, 4), dtype=np.uint8), None, None
    return get_camera_image


@pytest.fixture(autouse=True)
def _clean_look(monkeypatch):
    monkeypatch.delenv(generate_video.ENV_SCENARIO_RENDER, raising=False)
    generate_video._RENDER_LOOK.clear()
    yield
    generate_video._RENDER_LOOK.clear()


def test_without_the_switch_the_cameras_keep_the_fixed_light(monkeypatch):
    captured = {}
    monkeypatch.setattr(p, "getCameraImage", _fake_camera(captured))

    assert generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE) is None
    generate_video._render_rgb(0, None, None, 8, 6)

    assert captured["lightDirection"] == FIXED_LIGHT
    assert "hazeDistance" not in captured and "lightColor" not in captured


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_with_the_switch_the_cameras_take_the_scenarios_look(monkeypatch, value):
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, value)
    monkeypatch.setenv("SWARM_SCENARIO_WEATHER", "fog")
    monkeypatch.setenv("SWARM_SCENARIO_TIME", "dusk")
    captured = {}
    monkeypatch.setattr(p, "getCameraImage", _fake_camera(captured))

    spec = generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
    generate_video._render_rgb(0, None, None, 8, 6)

    assert (spec.weather, spec.time_of_day) == ("fog", "dusk")
    assert captured["lightDirection"] == list(spec.sun.direction) != FIXED_LIGHT
    assert captured["hazeDistance"] == spec.haze_m
    assert captured["skyHorizonColor"] == list(spec.sky()[0])


def test_the_look_is_reset_for_the_next_flight(monkeypatch):
    monkeypatch.setenv(generate_video.ENV_SCENARIO_RENDER, "1")
    generate_video._apply_scenario_look(SEED, CHALLENGE_TYPE)
    assert generate_video._RENDER_LOOK

    monkeypatch.delenv(generate_video.ENV_SCENARIO_RENDER)
    generate_video._apply_scenario_look(SEED + 1, CHALLENGE_TYPE)

    assert generate_video._RENDER_LOOK == {}
