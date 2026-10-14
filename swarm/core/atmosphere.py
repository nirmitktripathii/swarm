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

"""Haze and sky applied to a rendered frame in software, from its depth buffer.

Stock pybullet's `getCameraImage` rejects `hazeDistance` and the sky colours (only the swarm-bullet3
build takes them), so a video made on a plain install would crash or ignore the weather. This does the
same job on the finished frame: surfaces fade toward the fog colour with distance, and pixels no
surface covers take the sky gradient. The light itself (direction, colour, ambient, diffuse) is a
standard `getCameraImage` argument and does not go through here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

RGB = Tuple[float, float, float]

# getCameraImage arguments that only the swarm-bullet3 build accepts; the software path replaces them.
RENDERER_ONLY_KEYS = ("hazeDistance", "skyHorizonColor", "skyZenithColor")

_SKY_DEPTH = 1.0 - 1e-6  # a depth-buffer value at the far plane: no surface was drawn there


@dataclass(frozen=True)
class Atmosphere:
    """What the air does to a frame: how far it sees, what it fades to, and the sky behind."""

    haze_m: float
    fog_rgb: RGB
    sky: Optional[Tuple[RGB, RGB]]  # (horizon, zenith), or None to keep the renderer's own sky

    def __post_init__(self) -> None:
        if not self.haze_m > 0.0:
            raise ValueError(f"haze_m must be positive, got {self.haze_m!r}")


# Clear-air sky (horizon, zenith): the renderer's own background is flat white, so a clear day would
# look unlit next to a foggy one. Warm when the sun is low, blue when it is high; blended between.
_SKY_LOW_SUN = ((0.96, 0.74, 0.52), (0.36, 0.42, 0.66))
_SKY_HIGH_SUN = ((0.78, 0.88, 0.98), (0.26, 0.50, 0.90))
_LOW_SUN_DEG, _HIGH_SUN_DEG = 5.0, 35.0


def clear_sky(sun_elevation_deg: float) -> Tuple[RGB, RGB]:
    """(horizon, zenith) of a clear sky for a sun at this elevation."""
    t = float(np.clip((sun_elevation_deg - _LOW_SUN_DEG) / (_HIGH_SUN_DEG - _LOW_SUN_DEG), 0.0, 1.0))
    horizon, zenith = (
        tuple((1.0 - t) * lo + t * hi for lo, hi in zip(low, high))
        for low, high in zip(_SKY_LOW_SUN, _SKY_HIGH_SUN)
    )
    return horizon, zenith


def atmosphere_for(spec) -> Atmosphere:
    """The atmosphere of a `ScenarioSpec`; a clear day gets a clear sky instead of the renderer's white."""
    sky = spec.sky()
    if sky is None:
        sky = clear_sky(spec.sun.elevation_deg)
    return Atmosphere(haze_m=float(spec.haze_m), fog_rgb=tuple(spec.fog_rgb()), sky=sky)


def projection_planes(projection: Sequence[float]) -> Tuple[float, float]:
    """Near and far plane distances of an OpenGL perspective matrix given as 16 column-major numbers."""
    m = np.asarray(projection, dtype=float).reshape(-1)
    if m.size != 16:
        raise ValueError(f"projection matrix needs 16 values, got {m.size}")
    a, b = m[10], m[14]  # -(f+n)/(f-n) and -2fn/(f-n)
    return float(b / (a - 1.0)), float(b / (a + 1.0))


def depth_metres(zbuf: np.ndarray, near: float, far: float) -> np.ndarray:
    """Linear distance along the view axis, in metres, from a depth-buffer image in [0, 1]."""
    z = np.asarray(zbuf, dtype=np.float32)
    return far * near / np.maximum(far - (far - near) * z, near * 1e-6)


def apply_atmosphere(rgb: np.ndarray, zbuf: np.ndarray, near: float, far: float, atm: Atmosphere) -> np.ndarray:
    """Return a new (h, w, 3) uint8 frame with distance fog and the sky gradient applied.

    Fog follows Beer-Lambert: a surface `d` metres away keeps `exp(-d / haze_m)` of its own colour.
    Sky pixels (nothing drawn) get the horizon-to-zenith gradient, the horizon colour at the lowest
    sky row, when the atmosphere carries a sky; otherwise they keep the renderer's sky.
    """
    frame = np.asarray(rgb)
    h, w = frame.shape[:2]
    z = np.asarray(zbuf, dtype=np.float32).reshape(h, w)

    # Runs once per rendered frame, so it works in place on one (h, 3w) float32 buffer: broadcasting a
    # (h, w, 1) factor over the 3-wide colour axis is several times slower than a contiguous multiply.
    # keep = exp(-depth / haze) with depth = far * near / (far - (far - near) * z), as in depth_metres.
    denom = np.multiply(z, np.float32(-(far - near)))
    denom += np.float32(far)
    np.maximum(denom, np.float32(near * 1e-6), out=denom)
    keep = np.divide(np.float32(-far * near / atm.haze_m), denom)
    np.exp(keep, out=keep)

    fog = np.tile(np.asarray(atm.fog_rgb, dtype=np.float32) * 255.0, w)
    out = frame[..., :3].astype(np.float32).reshape(h, w * 3)
    out -= fog
    out *= np.repeat(keep, 3, axis=1)
    out += fog + np.float32(0.5)  # the half is the rounding: astype(uint8) truncates

    sky_mask = z >= _SKY_DEPTH
    if sky_mask.any():
        mask3 = np.repeat(sky_mask, 3, axis=1)
        if atm.sky is None:
            kept = frame[..., :3].reshape(h, w * 3).astype(np.float32) + np.float32(0.5)
            np.copyto(out, kept, where=mask3)
        else:
            horizon = np.asarray(atm.sky[0], dtype=np.float32) * 255.0
            zenith = np.asarray(atm.sky[1], dtype=np.float32) * 255.0
            lowest_sky_row = np.flatnonzero(sky_mask.any(axis=1)).max()
            t = np.clip(np.arange(h, dtype=np.float32) / max(float(lowest_sky_row), 1.0), 0.0, 1.0)[:, None]
            rows = zenith * (1.0 - t) + horizon * t + np.float32(0.5)
            np.copyto(out, np.tile(rows, (1, w)), where=mask3)

    return out.astype(np.uint8).reshape(h, w, 3)
