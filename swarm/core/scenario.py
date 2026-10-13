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

"""Seeded scenario conditions: season, time of day, weather, wind and air for one seed.

Every factor draws from its own random stream, so changing one factor (say the season)
leaves every other factor of the seed where it was. That is what makes paired comparisons
possible: the same seed under two seasons differs only by the season.

What each factor reaches, and through which channel:

    season       geometry   crop height and stage, flooded paddy, river width, leaf-off trees, snow
    time_of_day  light      a sun (or moon) for the renderer; depth observations do not change
    weather      light      sky colour and visibility (haze distance) for colour frames
                 physics    wind: steady speed, turbulence and gusts, fed to swarm.core.wind
    air          physics    site altitude and temperature give an air-density ratio (thrust scale)

India-first: seasons follow the Indian calendar (summer, monsoon, post-monsoon, winter) plus
a Himalayan snow season. A run can pin any factor through `sample_scenario(..., season=...)`
or the SWARM_SCENARIO_SEASON / SWARM_SCENARIO_TIME / SWARM_SCENARIO_WEATHER variables.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from swarm.constants import (
    MOON_AMBIENT_RANGE,
    MOON_COLOR,
    MOON_DIFFUSE_RANGE,
    SUN_AMBIENT_RANGE,
    SUN_DIFFUSE_MAX,
    SUN_EXTINCTION,
    SUN_MIN_ELEVATION_DEG,
)
from swarm.core.daylight import NIGHT_SKY, SunLight, _direction, max_elevation_deg, sun_color

RGB = Tuple[float, float, float]

# Own streams: no factor may ride another factor's draw.
SCENARIO_SEASON_SEED_OFFSET = 0x5EA50
SCENARIO_TIME_PICK_SEED_OFFSET = 0x71CE
SCENARIO_TIME_SEED_OFFSET = 0x71DE
SCENARIO_WEATHER_SEED_OFFSET = 0x3EA7E
SCENARIO_AIR_SEED_OFFSET = 0xA1B

SEASONS: Tuple[str, ...] = ("summer", "monsoon", "post_monsoon", "winter", "snow")
TIMES_OF_DAY: Tuple[str, ...] = ("dawn", "day", "dusk", "night")
WEATHERS: Tuple[str, ...] = ("clear", "haze", "fog", "rain", "dust", "snowfall")

SEASON_WEIGHTS: Mapping[str, float] = {
    "summer": 0.22, "monsoon": 0.24, "post_monsoon": 0.18, "winter": 0.24, "snow": 0.12,
}
TIME_WEIGHTS: Mapping[str, float] = {"dawn": 0.20, "day": 0.45, "dusk": 0.20, "night": 0.15}

# Weather that each season actually produces in India: dust storms in summer, rain in the
# monsoon, stubble-smoke haze after it, dense dawn fog in the Indo-Gangetic winter, snowfall
# only in the Himalayan snow season.
WEATHER_WEIGHTS_BY_SEASON: Mapping[str, Mapping[str, float]] = {
    "summer": {"clear": 0.45, "haze": 0.25, "dust": 0.25, "rain": 0.05},
    "monsoon": {"clear": 0.25, "rain": 0.55, "haze": 0.10, "fog": 0.10},
    "post_monsoon": {"clear": 0.30, "haze": 0.50, "fog": 0.15, "rain": 0.05},
    "winter": {"clear": 0.45, "fog": 0.35, "haze": 0.20},
    "snow": {"clear": 0.45, "fog": 0.25, "snowfall": 0.30},
}


@dataclass(frozen=True)
class WeatherModel:
    """Ranges one weather draws from: meteorological visibility, wind cap, turbulence, gust count."""

    visibility_m: Tuple[float, float]
    wind_max_mps: Tuple[float, float]
    turbulence: float
    gusts: Tuple[int, int]
    sky: Optional[Tuple[RGB, RGB]]  # (horizon, zenith); None keeps the renderer's own sky
    fog_rgb: RGB


WEATHER_MODELS: Mapping[str, WeatherModel] = {
    "clear": WeatherModel((15000.0, 30000.0), (1.0, 3.0), 0.5, (0, 1), None, (0.78, 0.84, 0.92)),
    "haze": WeatherModel((3000.0, 7000.0), (0.5, 2.0), 0.3, (0, 0),
                         ((0.86, 0.84, 0.78), (0.66, 0.72, 0.80)), (0.84, 0.82, 0.76)),
    "fog": WeatherModel((150.0, 800.0), (0.0, 1.0), 0.2, (0, 0),
                        ((0.80, 0.81, 0.82), (0.72, 0.74, 0.77)), (0.80, 0.81, 0.82)),
    "rain": WeatherModel((1500.0, 5000.0), (2.5, 4.5), 1.0, (1, 3),
                         ((0.52, 0.55, 0.58), (0.34, 0.37, 0.42)), (0.55, 0.58, 0.61)),
    "dust": WeatherModel((800.0, 3000.0), (3.0, 5.0), 1.0, (2, 3),
                         ((0.82, 0.68, 0.50), (0.70, 0.58, 0.44)), (0.80, 0.67, 0.50)),
    "snowfall": WeatherModel((400.0, 2000.0), (1.0, 3.0), 0.6, (0, 2),
                             ((0.86, 0.87, 0.90), (0.74, 0.76, 0.80)), (0.86, 0.87, 0.90)),
}

# Open ground funnels more wind than a sheltered forest floor; hills most of all.
WIND_EXPOSURE_BY_TYPE: Mapping[int, float] = {2: 1.2, 3: 1.35, 6: 0.6, 8: 1.2, 9: 1.2, 10: 1.35, 11: 1.25}
WIND_CAP_MPS = 6.0  # hard cap: a Crazyflie-class drone (3 m/s speed limit) cannot fly in more

# Site altitude bands (m) by challenge type; the snow season lifts hill sites to Himalayan heights.
SITE_ALTITUDE_BY_TYPE: Mapping[int, Tuple[float, float]] = {
    3: (900.0, 2600.0), 4: (900.0, 2600.0), 8: (50.0, 350.0), 9: (50.0, 350.0),
    10: (700.0, 2200.0), 11: (0.0, 60.0),
}
DEFAULT_SITE_ALTITUDE_M = (0.0, 300.0)
SNOW_SITE_ALTITUDE_M = (2200.0, 3500.0)
HILL_TYPES = (3, 4, 10)
# Sea-level air temperature by season; the snow season gives site values directly.
SEA_LEVEL_TEMP_C: Mapping[str, Tuple[float, float]] = {
    "summer": (36.0, 46.0), "monsoon": (26.0, 33.0), "post_monsoon": (20.0, 30.0),
    "winter": (4.0, 16.0), "snow": (-12.0, 2.0),
}
LAPSE_RATE_K_PER_M = 0.0065
ISA_T0_K = 288.15

# Visibility (5 % contrast) is about 3 / extinction; the renderer's haze distance is the
# 63 % point, 1 / extinction. So haze distance = visibility / 3.
VISIBILITY_TO_HAZE = 1.0 / 3.0

# Time-of-day sun bands. Azimuth is measured from +x (east) toward +y (north), so the
# southern sky sits at -90 degrees and the west at -180.
DAWN_DUSK_ELEVATION_DEG = (SUN_MIN_ELEVATION_DEG, 12.0)
DAY_MIN_ELEVATION_DEG = 25.0
DAY_HALF_SPAN_H = 4.0  # a 25 degree sun stands about 4 h from solar noon
DAWN_AZIMUTH_DEG = (-20.0, 20.0)
DUSK_AZIMUTH_DEG = (-200.0, -160.0)
MOON_ELEVATION_DEG = (15.0, 60.0)

ENV_SEASON = "SWARM_SCENARIO_SEASON"
ENV_TIME = "SWARM_SCENARIO_TIME"
ENV_WEATHER = "SWARM_SCENARIO_WEATHER"


@dataclass(frozen=True)
class ScenarioSpec:
    """All conditions of one seed, in the units the sim and the renderer take."""

    seed: int
    challenge_type: int
    season: str
    time_of_day: str
    weather: str
    visibility_m: float
    haze_m: float
    wind_max_mps: float
    wind_turbulence: float
    wind_gusts: int
    site_altitude_m: float
    air_temp_c: float
    air_density_ratio: float
    sun: SunLight

    def label(self) -> str:
        """One short line for overlays and logs."""
        return (f"{self.season.replace('_', '-')} | {self.time_of_day} | {self.weather} | "
                f"wind {self.wind_max_mps:.1f} m/s | {self.air_temp_c:.0f} C @ {self.site_altitude_m:.0f} m")

    def wind_task_fields(self) -> Dict[str, Any]:
        """The MapTask wind fields (see validator.task_gen._wind_for_map); empty when calm."""
        if self.wind_max_mps <= 0.0:
            return {}
        return {
            "wind_max_mps": self.wind_max_mps,
            "wind_turbulence": self.wind_turbulence,
            "wind_gusts": self.wind_gusts,
        }

    def sky(self) -> Optional[Tuple[RGB, RGB]]:
        """Sky (horizon, zenith): a moon's dark sky at night, else the weather's, else None."""
        if self.sun.night:
            return self.sun.sky
        return WEATHER_MODELS[self.weather].sky

    def fog_rgb(self) -> RGB:
        """Colour distant surfaces fade toward; dark at night."""
        if self.sun.night:
            return NIGHT_SKY[0]
        return WEATHER_MODELS[self.weather].fog_rgb

    def as_dict(self) -> Dict[str, Any]:
        """Plain dict for JSON logs."""
        return asdict(self)


def _stream(seed: int, offset: int) -> random.Random:
    return random.Random((int(seed) ^ int(offset)) & 0xFFFFFFFF)


def _weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    names = sorted(weights)  # fixed order: dict order must not change a draw
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]


def _check(name: str, value: Optional[str], allowed: Tuple[str, ...]) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"unknown {name} {value!r}; expected one of {allowed}")


def _daylight(elevation_deg: float, azimuth_deg: float, hour: float) -> SunLight:
    """A sun at an elevation, lit the way daylight.seeded_sun lights it."""
    peak = max_elevation_deg()
    air_mass = 1.0 / math.sin(math.radians(elevation_deg))
    diffuse = SUN_DIFFUSE_MAX * math.exp(-SUN_EXTINCTION * (air_mass - 1.0))
    lo, hi = SUN_AMBIENT_RANGE
    climb = (elevation_deg - SUN_MIN_ELEVATION_DEG) / (peak - SUN_MIN_ELEVATION_DEG)
    return SunLight(
        hour=round(hour, 3),
        elevation_deg=round(elevation_deg, 3),
        azimuth_deg=round(azimuth_deg, 3),
        direction=_direction(elevation_deg, azimuth_deg),
        color=sun_color(elevation_deg),
        ambient=round(lo + (hi - lo) * min(1.0, max(0.0, climb)), 4),
        diffuse=round(diffuse, 4),
        night=False,
        sky=None,
    )


def _moonlight(rng: random.Random) -> SunLight:
    elevation = rng.uniform(*MOON_ELEVATION_DEG)
    azimuth = rng.uniform(-180.0, 180.0)
    phase = rng.random()
    a_lo, a_hi = MOON_AMBIENT_RANGE
    d_lo, d_hi = MOON_DIFFUSE_RANGE
    return SunLight(
        hour=round(rng.uniform(20.0, 28.0) % 24.0, 3),
        elevation_deg=round(elevation, 3),
        azimuth_deg=round(azimuth, 3),
        direction=_direction(elevation, azimuth),
        color=MOON_COLOR,
        ambient=round(a_lo + (a_hi - a_lo) * phase, 4),
        diffuse=round(d_lo + (d_hi - d_lo) * phase, 4),
        night=True,
        sky=NIGHT_SKY,
    )


def sun_for_time(seed: int, time_of_day: str) -> SunLight:
    """The light for a time of day, drawn from the seed's own time stream."""
    _check("time_of_day", time_of_day, TIMES_OF_DAY)
    rng = _stream(seed, SCENARIO_TIME_SEED_OFFSET)
    if time_of_day == "night":
        return _moonlight(rng)
    if time_of_day == "dawn":
        elevation = rng.uniform(*DAWN_DUSK_ELEVATION_DEG)
        return _daylight(elevation, rng.uniform(*DAWN_AZIMUTH_DEG), 6.0 + elevation / 15.0)
    if time_of_day == "dusk":
        elevation = rng.uniform(*DAWN_DUSK_ELEVATION_DEG)
        return _daylight(elevation, rng.uniform(*DUSK_AZIMUTH_DEG), 18.0 - elevation / 15.0)
    peak = max_elevation_deg()
    elevation = rng.uniform(DAY_MIN_ELEVATION_DEG, peak)
    # Hours from noon shrink as the sun climbs; the sun swings east -> south -> west with them.
    from_noon = DAY_HALF_SPAN_H * (peak - elevation) / (peak - DAY_MIN_ELEVATION_DEG)
    hour = 12.0 - from_noon if rng.random() < 0.5 else 12.0 + from_noon
    azimuth = -90.0 + (12.0 - hour) / 6.0 * 90.0
    return _daylight(elevation, azimuth, hour)


def air_density_ratio(site_altitude_m: float, air_temp_c: float) -> float:
    """rho / rho0 against the ISA sea-level standard (15 C, 1013.25 hPa).

    Pressure falls with altitude (barometric formula), density falls with pressure and
    rises with cold. Rotor thrust at a fixed RPM scales with this ratio.
    """
    pressure_ratio = (1.0 - 2.25577e-5 * float(site_altitude_m)) ** 5.25588
    return pressure_ratio * ISA_T0_K / (273.15 + float(air_temp_c))


def scenario_overrides_from_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Optional[str]]:
    """Pinned factors from the environment; empty strings count as not pinned."""
    env = os.environ if environ is None else environ
    return {
        "season": env.get(ENV_SEASON) or None,
        "time_of_day": env.get(ENV_TIME) or None,
        "weather": env.get(ENV_WEATHER) or None,
    }


def _air(seed: int, challenge_type: int, season: str) -> Tuple[float, float]:
    rng = _stream(seed, SCENARIO_AIR_SEED_OFFSET)
    lo, hi = SITE_ALTITUDE_BY_TYPE.get(int(challenge_type), DEFAULT_SITE_ALTITUDE_M)
    if season == "snow" and int(challenge_type) in HILL_TYPES:
        lo, hi = SNOW_SITE_ALTITUDE_M
    altitude = rng.uniform(lo, hi)
    temp = rng.uniform(*SEA_LEVEL_TEMP_C[season])
    if season != "snow":
        temp -= LAPSE_RATE_K_PER_M * altitude
    return altitude, temp


def sample_scenario(
    seed: int,
    challenge_type: int,
    *,
    season: Optional[str] = None,
    time_of_day: Optional[str] = None,
    weather: Optional[str] = None,
) -> ScenarioSpec:
    """Draw every factor for a seed from its own stream; any factor passed in is pinned.

    A pinned weather the season would never produce (snowfall in the monsoon) is allowed: it
    is how stress cells are built on purpose.
    """
    _check("season", season, SEASONS)
    _check("time_of_day", time_of_day, TIMES_OF_DAY)
    _check("weather", weather, WEATHERS)

    season = season or _weighted(_stream(seed, SCENARIO_SEASON_SEED_OFFSET), SEASON_WEIGHTS)
    time_of_day = time_of_day or _weighted(_stream(seed, SCENARIO_TIME_PICK_SEED_OFFSET), TIME_WEIGHTS)
    sun = sun_for_time(seed, time_of_day)

    weather_rng = _stream(seed, SCENARIO_WEATHER_SEED_OFFSET)
    weather_pick = _weighted(weather_rng, WEATHER_WEIGHTS_BY_SEASON[season])
    weather = weather or weather_pick
    model = WEATHER_MODELS[weather]
    visibility = weather_rng.uniform(*model.visibility_m)
    exposure = WIND_EXPOSURE_BY_TYPE.get(int(challenge_type), 1.0)
    wind = min(WIND_CAP_MPS, weather_rng.uniform(*model.wind_max_mps) * exposure)
    gusts = weather_rng.randint(*model.gusts)

    altitude, temp = _air(seed, challenge_type, season)

    return ScenarioSpec(
        seed=int(seed),
        challenge_type=int(challenge_type),
        season=season,
        time_of_day=time_of_day,
        weather=weather,
        visibility_m=round(visibility, 1),
        haze_m=round(visibility * VISIBILITY_TO_HAZE, 1),
        wind_max_mps=round(wind, 3),
        wind_turbulence=model.turbulence,
        wind_gusts=int(gusts),
        site_altitude_m=round(altitude, 1),
        air_temp_c=round(temp, 2),
        air_density_ratio=round(air_density_ratio(altitude, temp), 4),
        sun=sun,
    )


def scenario_for_seed(seed: int, challenge_type: int) -> ScenarioSpec:
    """The scenario a map builder uses: the seed's draw with any factor pinned from the environment."""
    return sample_scenario(seed, challenge_type, **scenario_overrides_from_env())


__all__ = [
    "SEASONS", "TIMES_OF_DAY", "WEATHERS", "WEATHER_MODELS", "ScenarioSpec", "WeatherModel",
    "air_density_ratio", "sample_scenario", "scenario_for_seed", "scenario_overrides_from_env",
    "sun_for_time",
]
