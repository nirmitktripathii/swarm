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

"""Would a time-of-flight sensor have seen this wire? A per-pixel detection model for thin conductors.

The rasteriser is the wrong judge of wire visibility. A 4 mm conductor is far below one pixel of a
128 px depth image, so the renderer draws it in a few lucky pixels or not at all, while the physics
always collides with it. Real ranging sensors do see wires, but only when the wire fills enough of
a pixel's footprint and returns enough light. This module replaces the renderer's luck with that
physics:

    fill  = min(1, d / (R * ifov * min_pixels))       fraction of the pixel footprint the wire covers
    snr   = (fill * rho / 0.10) * (R_ref / R)^2 * exp(-2 * sigma * R)
    sigma = 3.912 / visibility                         Koschmieder extinction, two-way path
    P     = snr^k / (1 + snr^k)                         detection probability, 0.5 at snr = 1

R_ref is the range at which the sensor just detects (P = 0.5) a 10 %-reflectance target filling
its pixel, which is how datasheets quote range. min_pixels > 1 models stereo depth, whose
matching window must see the wire across several pixels before it yields a disparity.

Wires come in as the WireSpan list the biome builders record, are projected into the camera with
the same view and projection matrices the renderer used, and then, per pixel along each wire,
either written into the depth image (detected) or, where the renderer happened to draw them,
painted over with the background from the same column (missed). Everything is numpy; the
pybullet render is untouched. The model is opt-in (SWARM_TOF_WIRE_MODEL) and off by default.

Sensor presets are representative of each class from public datasheet figures; they are not a
calibration of any unit and are labelled as assumptions where the datasheet is silent.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

TOF_ENV_VAR = "SWARM_TOF_WIRE_MODEL"
REFERENCE_REFLECTANCE = 0.10
KOSCHMIEDER = 3.912          # ln(1/0.02): visibility is the range at 2 % contrast
DEFAULT_REFLECTANCE = 0.20
DEPTH_MATCH_TOL_M = 0.05     # a rendered pixel within this (or 2 %) of the wire depth is the wire itself
SAMPLES_PER_PIXEL = 2.0

# Diffuse NIR reflectance by conductor kind (assumed): bare aluminium/steel versus black insulated LT.
REFLECTANCE_BY_KIND: Dict[str, float] = {
    "11kv": 0.35, "ht": 0.35, "earth": 0.30, "stay": 0.30, "lt": 0.08, "service": 0.08,
}


@dataclass(frozen=True)
class TofSensor:
    """A ranging sensor class: pixel footprint, datasheet range at 10 % reflectance, curve steepness."""

    name: str
    ifov_rad: float
    range_ref_m: float
    min_range_m: float = 0.05
    min_pixels: float = 1.0
    steepness: float = 4.0


SENSORS: Dict[str, TofSensor] = {
    # 8x8 multizone ToF, 60 deg square FoV: 7.5 deg zones. Range at 10 % assumed ~1.7 m (datasheet: 3.5 m max).
    "vl53l7cx": TofSensor("vl53l7cx", math.radians(60.0 / 8.0), 1.7, min_range_m=0.02),
    # Single-point lidar, 3.6 deg FoV, 4 m at 10 % reflectance (datasheet).
    "tfmini_plus": TofSensor("tfmini_plus", math.radians(3.6), 4.0, min_range_m=0.1),
    # Scanning lidar, 40 m at 10 % (datasheet); beam footprint ~0.27 deg assumed.
    "livox_mid360": TofSensor("livox_mid360", math.radians(0.27), 40.0, min_range_m=0.1),
    # Active stereo depth camera, 87 deg over 848 px; a 7 px matching window must span the wire (assumed).
    "depth_cam": TofSensor("depth_cam", math.radians(87.0 / 848.0), 6.0, min_range_m=0.2, min_pixels=7.0),
}


def sensor_from_env(environ: Optional[Dict[str, str]] = None) -> Optional[TofSensor]:
    """The sensor named by SWARM_TOF_WIRE_MODEL, or None when it is unset, empty, '0' or 'off'."""
    env = os.environ if environ is None else environ
    name = env.get(TOF_ENV_VAR, "").strip().lower()
    if name in ("", "0", "off", "none"):
        return None
    if name not in SENSORS:
        raise ValueError(f"{TOF_ENV_VAR}={name!r}: expected one of {sorted(SENSORS)} or 'off'")
    return SENSORS[name]


# -- which wires are in which world -----------------------------------------------------------
# The map dispatcher records every build here, wires or none, so a reset never sees the last map's spans.
_WIRES_BY_CLIENT: Dict[int, Tuple] = {}


def register_wires(cli: int, spans: Sequence) -> None:
    _WIRES_BY_CLIENT[int(cli)] = tuple(spans)


def wires_for(cli: int) -> Tuple:
    return _WIRES_BY_CLIENT.get(int(cli), ())


# -- the detection physics --------------------------------------------------------------
def extinction_per_m(visibility_m: float) -> float:
    return KOSCHMIEDER / max(float(visibility_m), 1.0)


def fill_fraction(diameter_m, range_m, sensor: TofSensor):
    """Fraction of one pixel's footprint a wire of this diameter covers at this range."""
    footprint = np.maximum(np.asarray(range_m, dtype=np.float64), 1e-6) * sensor.ifov_rad * sensor.min_pixels
    return np.minimum(1.0, np.asarray(diameter_m, dtype=np.float64) / footprint)


def detection_probability(range_m, diameter_m, reflectance, sensor: TofSensor, visibility_m: float = 20000.0):
    """Probability the sensor returns a range for a wire pixel (vectorised over range/diameter/reflectance)."""
    r = np.maximum(np.asarray(range_m, dtype=np.float64), 1e-6)
    fill = fill_fraction(diameter_m, r, sensor)
    snr = (fill * np.asarray(reflectance, dtype=np.float64) / REFERENCE_REFLECTANCE) \
        * (sensor.range_ref_m / r) ** 2 * np.exp(-2.0 * extinction_per_m(visibility_m) * r)
    snr_k = snr ** sensor.steepness
    p = snr_k / (1.0 + snr_k)
    return np.where(r < sensor.min_range_m, 0.0, p)


# -- depth buffer conversions (pybullet's OpenGL z-buffer) -----------------------------------
def buffer_to_metric(buf: np.ndarray, near: float, far: float) -> np.ndarray:
    d = np.clip(np.asarray(buf, dtype=np.float64), 0.0, 1.0)
    return far * near / np.maximum(far - (far - near) * d, near * 1e-6)


def metric_to_buffer(z: np.ndarray, near: float, far: float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), near, far)
    return far * (z - near) / (z * (far - near))


# -- projecting wires into the image -------------------------------------------------------
def _matrix(m: Sequence[float]) -> np.ndarray:
    """pybullet's column-major 16-list as a row-major 4x4."""
    return np.asarray(m, dtype=np.float64).reshape(4, 4).T


def _transform(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Homogeneous 4x4 transform of (N, 3) points. einsum, not matmul: a 4x4 needs no BLAS, and some
    MKL builds crash in dgemm when their OpenMP runtime is mismatched."""
    return np.einsum("ij,nj->ni", m, np.c_[pts, np.ones(len(pts))])


def _clip_near(a: np.ndarray, b: np.ndarray, near: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip an eye-space segment to the near plane (eye looks down -z); None if wholly behind."""
    za, zb = -a[2], -b[2]
    lim = near * 1.001
    if za < lim and zb < lim:
        return None
    if za < lim:
        a = a + (b - a) * (lim - za) / (zb - za)
    elif zb < lim:
        b = b + (a - b) * (lim - zb) / (za - zb)
    return a, b


def project_wires(spans: Sequence, view: Sequence[float], proj: Sequence[float], width: int, height: int,
                  near: float, far: float, n_segments: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-wire eye depth, diameter and reflectance per pixel (inf / 0 where no wire crosses).

    Each span is sampled along its catenary, every segment clipped to the near plane and stepped
    at SAMPLES_PER_PIXEL samples per pixel of image length, so a wire leaves an unbroken line.
    """
    V, P = _matrix(view), _matrix(proj)
    idx_all, dep_all, dia_all, ref_all = [], [], [], []
    for span in spans:
        pts = np.asarray(span.points(n_segments), dtype=np.float64)
        eye = _transform(V, pts)[:, :3]
        for a, b in zip(eye[:-1], eye[1:]):
            seg = _clip_near(a, b, near)
            if seg is None:
                continue
            a, b = seg
            ends = _transform(P, np.vstack([a, b]))
            ndc = ends[:, :2] / ends[:, 3:4]
            px_len = math.hypot((ndc[1, 0] - ndc[0, 0]) * width / 2.0, (ndc[1, 1] - ndc[0, 1]) * height / 2.0)
            n = int(min(4096, math.ceil(px_len * SAMPLES_PER_PIXEL))) + 2
            t = np.linspace(0.0, 1.0, n)[:, None]
            samples = a + (b - a) * t                    # interpolate in eye space: perspective-correct
            clip = _transform(P, samples)
            u = ((clip[:, 0] / clip[:, 3]) + 1.0) * 0.5 * width
            v = (1.0 - (clip[:, 1] / clip[:, 3])) * 0.5 * height   # image row 0 is the top
            depth = -samples[:, 2]
            ok = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (depth <= far)
            if not ok.any():
                continue
            idx_all.append(v[ok].astype(np.int64) * width + u[ok].astype(np.int64))
            dep_all.append(depth[ok])
            dia_all.append(np.full(int(ok.sum()), 2.0 * span.radius_m))
            ref_all.append(np.full(int(ok.sum()), REFLECTANCE_BY_KIND.get(span.kind, DEFAULT_REFLECTANCE)))
    wire_depth = np.full(height * width, np.inf)
    wire_dia = np.zeros(height * width)
    wire_ref = np.zeros(height * width)
    if idx_all:
        idx, dep = np.concatenate(idx_all), np.concatenate(dep_all)
        dia, ref = np.concatenate(dia_all), np.concatenate(ref_all)
        order = np.lexsort((dep, idx))                 # by pixel, nearest first
        first = np.r_[True, idx[order][1:] != idx[order][:-1]]
        keep = order[first]
        wire_depth[idx[keep]] = dep[keep]
        wire_dia[idx[keep]] = dia[keep]
        wire_ref[idx[keep]] = ref[keep]
    shape = (height, width)
    return wire_depth.reshape(shape), wire_dia.reshape(shape), wire_ref.reshape(shape)


# -- applying detections and misses ---------------------------------------------------------
def apply_wire_miss(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint masked pixels with the nearest unmasked pixel in the same column (the background behind
    a missed wire); a column with nothing unmasked keeps its values."""
    out = np.array(depth_m, dtype=np.float64, copy=True)
    if not mask.any():
        return out
    h = out.shape[0]
    rows = np.arange(h)[:, None]
    valid = ~mask
    above = np.where(valid, rows, -1)
    np.maximum.accumulate(above, axis=0, out=above)            # last valid row at or above
    below = np.where(valid, rows, h)
    below = np.minimum.accumulate(below[::-1], axis=0)[::-1]   # first valid row at or below
    use_above = (above >= 0) & ((below >= h) | (rows - above <= below - rows))
    src = np.where(use_above, above, below)
    has_src = (src >= 0) & (src < h)
    cols = np.broadcast_to(np.arange(out.shape[1]), out.shape)
    fill = mask & has_src
    out[fill] = out[src[fill], cols[fill]]
    return out


def apply_tof_wire_model(depth_buffer: np.ndarray, wire_depth: np.ndarray, wire_dia: np.ndarray,
                         wire_ref: np.ndarray, sensor: TofSensor, rng: np.random.Generator, *,
                         near: float, far: float, visibility_m: float = 20000.0) -> np.ndarray:
    """The z-buffer a sensor of this class would have produced given the wires in view.

    Detected wire pixels take the wire's depth (unless something nearer was rendered there); wire
    pixels the renderer drew but the sensor missed are painted with the background.
    """
    metric = buffer_to_metric(depth_buffer, near, far)
    wire_px = np.isfinite(wire_depth)
    if not wire_px.any():
        return np.asarray(depth_buffer, dtype=np.float32)
    tol = np.maximum(DEPTH_MATCH_TOL_M, 0.02 * np.where(wire_px, wire_depth, 0.0))
    visible = wire_px & (wire_depth <= metric + tol)           # not hidden behind nearer geometry
    drawn = visible & (np.abs(metric - np.where(wire_px, wire_depth, 0.0)) <= tol)
    p = detection_probability(np.where(wire_px, wire_depth, 1.0), wire_dia, wire_ref, sensor, visibility_m)
    detected = visible & (rng.random(metric.shape) < p)
    out = apply_wire_miss(metric, drawn & ~detected)
    out[detected] = np.minimum(out[detected], wire_depth[detected])
    return metric_to_buffer(out, near, far).astype(np.float32)


__all__ = [
    "SENSORS", "TOF_ENV_VAR", "TofSensor", "apply_tof_wire_model", "apply_wire_miss", "buffer_to_metric",
    "detection_probability", "fill_fraction", "metric_to_buffer", "project_wires", "register_wires",
    "sensor_from_env", "wires_for",
]
