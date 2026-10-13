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


def atmosphere_for(spec) -> Atmosphere:
    """The atmosphere of a `ScenarioSpec`."""
    return Atmosphere(haze_m=float(spec.haze_m), fog_rgb=tuple(spec.fog_rgb()), sky=spec.sky())


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
    base = frame[..., :3].astype(np.float32) / 255.0

    keep = np.exp(-depth_metres(z, near, far) / atm.haze_m)[..., None]
    out = base * keep + np.asarray(atm.fog_rgb, dtype=np.float32) * (1.0 - keep)

    sky_mask = (z >= _SKY_DEPTH)[..., None]
    if atm.sky is None:
        out = np.where(sky_mask, base, out)
    elif sky_mask.any():
        horizon = np.asarray(atm.sky[0], dtype=np.float32)
        zenith = np.asarray(atm.sky[1], dtype=np.float32)
        lowest_sky_row = np.flatnonzero(sky_mask[..., 0].any(axis=1)).max()
        t = np.clip(np.arange(h, dtype=np.float32) / max(float(lowest_sky_row), 1.0), 0.0, 1.0)[:, None, None]
        gradient = np.broadcast_to(zenith * (1.0 - t) + horizon * t, out.shape)
        out = np.where(sky_mask, gradient, out)

    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
