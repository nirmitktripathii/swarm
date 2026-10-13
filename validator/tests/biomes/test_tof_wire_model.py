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

"""The ToF wire model: detection physics, the env switch, the projection against a real render, and the rewrite."""
from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from swarm.core.maps.biome_kit import Kit, WireSpan
from swarm.core.tof_wire_model import (
    SENSORS,
    TOF_ENV_VAR,
    apply_tof_wire_model,
    apply_wire_miss,
    buffer_to_metric,
    detection_probability,
    metric_to_buffer,
    project_wires,
    register_wires,
    sensor_from_env,
    wires_for,
)

NEAR, FAR = 0.05, 20.0
CLEAR_AIR_M = 1e12


# -- detection physics -------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SENSORS))
def test_detection_falls_with_range_and_rises_with_diameter(name):
    sensor = SENSORS[name]
    ranges = np.linspace(max(sensor.min_range_m, 0.05) * 1.01, 60.0, 200)
    p_thin = detection_probability(ranges, 0.004, 0.08, sensor)
    p_thick = detection_probability(ranges, 0.024, 0.08, sensor)
    assert np.all(np.diff(p_thin) <= 1e-12)
    assert np.all(p_thick >= p_thin - 1e-12)


def test_fog_only_ever_lowers_detection():
    sensor = SENSORS["livox_mid360"]
    ranges = np.linspace(1.0, 40.0, 50)
    clear = detection_probability(ranges, 0.012, 0.35, sensor, visibility_m=CLEAR_AIR_M)
    fog = detection_probability(ranges, 0.012, 0.35, sensor, visibility_m=200.0)
    assert np.all(fog <= clear) and fog[-1] < clear[-1]


def test_datasheet_point_is_the_half_probability_point():
    """A 10 % target filling the pixel at the datasheet range is detected half the time."""
    sensor = SENSORS["tfmini_plus"]
    fill_dia = sensor.range_ref_m * sensor.ifov_rad * sensor.min_pixels
    assert detection_probability(sensor.range_ref_m, fill_dia, 0.10, sensor, CLEAR_AIR_M) == pytest.approx(0.5)


def test_nothing_is_ranged_inside_the_minimum_range():
    sensor = SENSORS["depth_cam"]
    assert detection_probability(sensor.min_range_m * 0.5, 0.05, 0.9, sensor) == 0.0


def test_a_scanning_lidar_sees_a_service_drop_a_multizone_tof_cannot():
    """The headline result: at 10 m a 6 mm black drop is usually a lidar return (snr ~1.6, P ~0.87)
    and never an 8x8 ToF one."""
    lidar = float(detection_probability(10.0, 0.006, 0.08, SENSORS["livox_mid360"]))
    zone_tof = float(detection_probability(10.0, 0.006, 0.08, SENSORS["vl53l7cx"]))
    assert lidar > 0.8 and zone_tof < 1e-3


# -- the env switch ------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["", "0", "off", "None", "  OFF "])
def test_switch_off_values_disable_the_model(value):
    assert sensor_from_env({TOF_ENV_VAR: value}) is None
    assert sensor_from_env({}) is None


def test_switch_selects_a_preset_and_rejects_unknown_names():
    assert sensor_from_env({TOF_ENV_VAR: "Livox_Mid360"}) is SENSORS["livox_mid360"]
    with pytest.raises(ValueError, match="expected one of"):
        sensor_from_env({TOF_ENV_VAR: "radar"})


def test_wire_registry_is_per_client_and_replaced_on_each_build():
    span = WireSpan((0, 0, 5), (10, 0, 5), 0.2, 0.004, "lt")
    register_wires(901, [span])
    register_wires(902, [])
    assert wires_for(901) == (span,) and wires_for(902) == () and wires_for(903) == ()
    register_wires(901, ())
    assert wires_for(901) == ()


# -- buffers and inpainting --------------------------------------------------------------------
def test_depth_buffer_round_trips_through_metres():
    z = np.array([0.05, 0.3, 2.0, 9.9, 20.0])
    assert buffer_to_metric(metric_to_buffer(z, NEAR, FAR), NEAR, FAR) == pytest.approx(z, rel=1e-9)


def test_missed_wire_pixels_take_the_nearest_background_in_their_column():
    depth = np.array([[10.0, 7.0], [5.0, 5.0], [12.0, 5.0], [14.0, 5.0]])
    mask = np.array([[False, True], [True, True], [True, True], [False, True]])
    out = apply_wire_miss(depth, mask)
    assert out[1, 0] == 10.0 and out[2, 0] == 14.0          # nearer row wins; ties go up
    assert np.array_equal(out[:, 1], depth[:, 1])          # nothing unmasked in the column: unchanged
    assert depth[1, 0] == 5.0                              # input left alone


# -- projection against pybullet's own render ------------------------------------------------
def _render(cli, view, proj, w, h):
    img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                           flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=cli)
    return np.reshape(np.asarray(img[3], dtype=np.float32), (h, w))


def test_projected_wire_lands_where_the_renderer_draws_it(cli):
    """A thick sloping wire in a non-square image: same rows, same columns, same depth (to its radius)."""
    w, h = 96, 64
    span = WireSpan((6.0, -4.0, 2.6), (6.0, 2.0, 1.1), 0.3, 0.15, "lt")
    Kit(cli, [], 0.0).wire(span, (0.1, 0.1, 0.1, 1.0), n_segments=16)
    view = p.computeViewMatrix([0, 0, 1.0], [20, 0, 1.0], [0, 0, 1], physicsClientId=cli)
    proj = p.computeProjectionMatrixFOV(70.0, w / h, NEAR, FAR, physicsClientId=cli)
    metric = buffer_to_metric(_render(cli, view, proj, w, h), NEAR, FAR)
    rendered = metric < FAR * 0.99

    wire_depth, _, _ = project_wires([span], view, proj, w, h, NEAR, FAR, n_segments=16)
    projected = np.isfinite(wire_depth)

    assert projected.sum() > 20
    assert (rendered & projected).sum() / projected.sum() > 0.9
    ry, rx = np.nonzero(rendered)
    py_, px_ = np.nonzero(projected)
    assert abs(ry.mean() - py_.mean()) < 1.5 and abs(rx.mean() - px_.mean()) < 1.5
    assert px_[py_.argmin()] == pytest.approx(rx[ry.argmin()], abs=1)   # the high end is on the same side
    both = rendered & projected
    assert np.median(np.abs(metric[both] - wire_depth[both])) < 0.2     # surface vs centreline: one radius


def test_wires_behind_the_camera_project_nowhere(cli):
    view = p.computeViewMatrix([0, 0, 1.0], [20, 0, 1.0], [0, 0, 1], physicsClientId=cli)
    proj = p.computeProjectionMatrixFOV(70.0, 1.0, NEAR, FAR, physicsClientId=cli)
    span = WireSpan((-6.0, -4.0, 1.0), (-6.0, 4.0, 1.0), 0.2, 0.004, "lt")
    wire_depth, _, _ = project_wires([span], view, proj, 32, 32, NEAR, FAR)
    assert not np.isfinite(wire_depth).any()


# -- the rewrite ----------------------------------------------------------------------------
def _scene(wire_m, background_m, drawn):
    """An 8x8 buffer at the background depth with a wire across row 3, drawn by the renderer or not."""
    metric = np.full((8, 8), background_m)
    if drawn:
        metric[3, :] = wire_m
    wire_depth = np.full((8, 8), np.inf)
    wire_depth[3, :] = wire_m
    return metric_to_buffer(metric, NEAR, FAR).astype(np.float32), wire_depth


def _rewrite(buffer, wire_depth, dia, refl, sensor, visibility_m=CLEAR_AIR_M):
    out = apply_tof_wire_model(buffer, wire_depth, np.full_like(wire_depth, dia), np.full_like(wire_depth, refl),
                               SENSORS[sensor], np.random.default_rng(0), near=NEAR, far=FAR,
                               visibility_m=visibility_m)
    assert out.dtype == np.float32 and out.shape == buffer.shape
    return buffer_to_metric(out, NEAR, FAR)


def test_a_wire_the_sensor_would_miss_is_erased_even_where_the_renderer_drew_it():
    buffer, wire_depth = _scene(wire_m=12.0, background_m=18.0, drawn=True)
    out = _rewrite(buffer, wire_depth, dia=0.004, refl=0.08, sensor="vl53l7cx")
    assert out[3] == pytest.approx(np.full(8, 18.0), rel=1e-4)


def test_a_wire_the_sensor_would_see_is_written_in_even_where_the_renderer_lost_it():
    buffer, wire_depth = _scene(wire_m=5.0, background_m=18.0, drawn=False)
    out = _rewrite(buffer, wire_depth, dia=0.024, refl=0.35, sensor="livox_mid360")
    assert out[3] == pytest.approx(np.full(8, 5.0), rel=1e-4)
    assert np.delete(out, 3, axis=0) == pytest.approx(np.full((7, 8), 18.0), rel=1e-4)


def test_a_wire_behind_a_wall_stays_hidden():
    buffer, wire_depth = _scene(wire_m=12.0, background_m=4.0, drawn=False)
    out = _rewrite(buffer, wire_depth, dia=0.024, refl=0.35, sensor="livox_mid360")
    assert out == pytest.approx(np.full((8, 8), 4.0), rel=1e-4)


def test_no_wires_in_view_returns_the_buffer_unchanged():
    buffer, wire_depth = _scene(wire_m=5.0, background_m=18.0, drawn=False)
    wire_depth[:] = np.inf
    out = apply_tof_wire_model(buffer, wire_depth, np.zeros_like(wire_depth), np.zeros_like(wire_depth),
                               SENSORS["depth_cam"], np.random.default_rng(0), near=NEAR, far=FAR)
    assert np.array_equal(out, buffer)
