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

"""Shared primitives for the biome maps (types 8-11): shapes, trees, clearings and wire spans.

The pure helpers (intervals, catenary points, clearings, colour tints) need no physics client
and are unit-tested on their own. `Kit` wraps one PyBullet client: it caches shapes so equal
bodies share one shape, keeps every body clear of the start and goal, and sorts what it builds
into trees, props, beams (wires) and terrain for the SAR tagger.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]
Interval = Tuple[float, float]

GROUND_HALF_THICKNESS_M = 0.05  # same slab as the forest floor: top face at z = 0
CLEARING_MIN_M = 8.0            # every map keeps this much open ground around start and goal
MAP_HALF_EXTENT_M = 60.0        # built area; the playable range (+-42 m) sits inside it
WIRE_COLLISION_RADIUS_MIN_M = 0.02  # thinner collision shapes tunnel at 50 Hz with a 3 m/s drone


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def subtract_intervals(lo: float, hi: float, holes: Iterable[Interval], min_len: float = 0.0) -> List[Interval]:
    """Pieces of [lo, hi] left after removing every hole, dropping pieces shorter than min_len."""
    pieces: List[Interval] = [(lo, hi)]
    for h_lo, h_hi in sorted(holes):
        nxt: List[Interval] = []
        for a, b in pieces:
            if h_hi <= a or h_lo >= b:
                nxt.append((a, b))
                continue
            if h_lo > a:
                nxt.append((a, h_lo))
            if h_hi < b:
                nxt.append((h_hi, b))
        pieces = nxt
    return [(a, b) for a, b in pieces if b - a >= max(min_len, 1e-9)]


def chord_hole(line_coord: float, centre_along: float, centre_across: float, radius: float) -> Optional[Interval]:
    """The stretch of a straight line (across = line_coord) that lies inside a circle, or None."""
    d = abs(line_coord - centre_across)
    if d >= radius:
        return None
    half = math.sqrt(radius * radius - d * d)
    return (centre_along - half, centre_along + half)


def catenary_points(a: Vec3, b: Vec3, sag_m: float, n_segments: int) -> List[Vec3]:
    """Points along a hanging wire from a to b with a parabolic sag of sag_m at mid-span.

        z(t) = za + (zb - za) t - 4 sag t (1 - t),   t in [0, 1]

    A parabola is within 1 % of a true catenary for the sag-to-span ratios of distribution lines.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1")
    pts: List[Vec3] = []
    for i in range(n_segments + 1):
        t = i / n_segments
        pts.append((
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t - 4.0 * sag_m * t * (1.0 - t),
        ))
    return pts


def is_clear(x: float, y: float, safe_zones: Sequence[Tuple[float, float]], clear_radius: float,
             footprint: float = 0.0) -> bool:
    """True when a footprint of radius `footprint` at (x, y) stays outside every clearing."""
    reach = clear_radius + footprint
    return all((x - sx) ** 2 + (y - sy) ** 2 >= reach * reach for sx, sy in safe_zones)


def segment_clear(a: Tuple[float, float], b: Tuple[float, float], safe_zones: Sequence[Tuple[float, float]],
                  clear_radius: float) -> bool:
    """True when the straight segment a-b passes outside every clearing."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    for sx, sy in safe_zones:
        t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0, ((sx - ax) * dx + (sy - ay) * dy) / length_sq))
        px, py = ax + t * dx, ay + t * dy
        if (px - sx) ** 2 + (py - sy) ** 2 < clear_radius * clear_radius:
            return False
    return True


def tint(rgba: Sequence[float], factor: float) -> RGBA:
    """A colour scaled by factor (clamped to [0, 1]), alpha unchanged."""
    r, g, b = (max(0.0, min(1.0, c * factor)) for c in rgba[:3])
    return (r, g, b, float(rgba[3]) if len(rgba) > 3 else 1.0)


def quat_z_to(direction: Vec3) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) turning +z onto `direction`."""
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    if dz > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if dz < -1.0 + 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay = -dy, dx  # z x d
    s = math.sqrt(ax * ax + ay * ay)
    half = math.acos(max(-1.0, min(1.0, dz))) / 2.0
    k = math.sin(half) / s
    return (ax * k, ay * k, 0.0, math.cos(half))


def quat_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a rotation of yaw radians about +z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class Frame:
    """A local map frame turned by yaw about the origin: builders lay features along +x, then turn them."""

    def __init__(self, yaw: float):
        self.yaw, self.c, self.s = float(yaw), math.cos(yaw), math.sin(yaw)

    def w(self, lx: float, ly: float) -> Tuple[float, float]:
        """Local (lx, ly) to world (x, y)."""
        return self.c * lx - self.s * ly, self.s * lx + self.c * ly

    def w3(self, lx: float, ly: float, z: float) -> Vec3:
        x, y = self.w(lx, ly)
        return x, y, z

    def local(self, x: float, y: float) -> Tuple[float, float]:
        """World (x, y) to local (lx, ly)."""
        return self.c * x + self.s * y, -self.s * x + self.c * y


def biome_cache_dir(name: str) -> str:
    """A writable folder for generated meshes and textures of one biome."""
    root = os.environ.get("SWARM_BIOME_CACHE") or os.path.join(tempfile.gettempdir(), "swarm_biomes")
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Wires
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WireSpan:
    """One conductor between two attachment points, with the sag it was built with."""

    a: Vec3
    b: Vec3
    sag_m: float
    radius_m: float
    kind: str  # "lt", "11kv", "ht", "stay", "service"

    def points(self, n_segments: int = 8) -> List[Vec3]:
        """The polyline the wire was built from."""
        return catenary_points(self.a, self.b, self.sag_m, n_segments)


# ---------------------------------------------------------------------------
# PyBullet kit
# ---------------------------------------------------------------------------
@dataclass
class BuiltMap:
    """What a biome builder made, sorted for the SAR tagger, plus a summary for logs."""

    trees: List[int] = field(default_factory=list)
    props: List[int] = field(default_factory=list)
    beams: List[int] = field(default_factory=list)
    terrain: List[int] = field(default_factory=list)
    wires: List[WireSpan] = field(default_factory=list)
    info: Dict[str, object] = field(default_factory=dict)

    def as_assets(self) -> Dict[str, object]:
        """The dict the map dispatcher reads (forest-builder compatible keys plus our own)."""
        return {
            "trees": list(self.trees), "props": list(self.props), "beams": list(self.beams),
            "terrain": list(self.terrain), "wires": list(self.wires), "info": dict(self.info),
        }


class Kit:
    """Shape factory on one PyBullet client that respects the clearings around start and goal."""

    def __init__(self, cli: int, safe_zones: Sequence[Tuple[float, float]], clear_radius: float):
        import pybullet as p  # imported here so the pure helpers above need no physics engine

        self.p = p
        self.cli = int(cli)
        self.safe_zones = [(float(x), float(y)) for x, y in safe_zones]
        self.clear_radius = max(float(clear_radius), CLEARING_MIN_M)
        self.out = BuiltMap()
        self._shapes: Dict[tuple, Tuple[int, int]] = {}

    # -- placement guards -------------------------------------------------
    def clear(self, x: float, y: float, footprint: float = 0.0) -> bool:
        """True when a footprint at (x, y) stays outside every clearing."""
        return is_clear(x, y, self.safe_zones, self.clear_radius, footprint)

    def segment_clear(self, a: Tuple[float, float], b: Tuple[float, float], margin: float = 0.0) -> bool:
        """True when a straight run a-b passes outside every clearing (plus margin)."""
        return segment_clear(a, b, self.safe_zones, self.clear_radius + margin)

    # -- shapes ---------------------------------------------------------------
    def _shape(self, kind: str, dims: Tuple[float, ...], rgba: Sequence[float], collide: bool,
               collision_dims: Optional[Tuple[float, ...]] = None) -> Tuple[int, int]:
        key = (kind, tuple(round(d, 4) for d in dims), tuple(round(c, 3) for c in rgba), collide,
               None if collision_dims is None else tuple(round(d, 4) for d in collision_dims))
        if key in self._shapes:
            return self._shapes[key]
        p, cli = self.p, self.cli
        cdims = collision_dims or dims
        if kind == "box":
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(dims), rgbaColor=list(rgba),
                                      specularColor=[0, 0, 0], physicsClientId=cli)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(cdims), physicsClientId=cli) if collide else -1
        elif kind == "cylinder":
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=dims[0], length=dims[1], rgbaColor=list(rgba),
                                      specularColor=[0, 0, 0], physicsClientId=cli)
            col = (p.createCollisionShape(p.GEOM_CYLINDER, radius=cdims[0], height=cdims[1], physicsClientId=cli)
                   if collide else -1)
        elif kind == "capsule":
            vis = p.createVisualShape(p.GEOM_CAPSULE, radius=dims[0], length=dims[1], rgbaColor=list(rgba),
                                      specularColor=[0, 0, 0], physicsClientId=cli)
            col = (p.createCollisionShape(p.GEOM_CAPSULE, radius=cdims[0], height=cdims[1], physicsClientId=cli)
                   if collide else -1)
        elif kind == "sphere":
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=dims[0], rgbaColor=list(rgba),
                                      specularColor=[0, 0, 0], physicsClientId=cli)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=cdims[0], physicsClientId=cli) if collide else -1
        else:
            raise ValueError(f"unknown shape kind {kind!r}")
        self._shapes[key] = (vis, col)
        return vis, col

    def _body(self, shapes: Tuple[int, int], pos: Sequence[float], orn=(0.0, 0.0, 0.0, 1.0), bucket: str = "props") -> int:
        vis, col = shapes
        uid = self.p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                     basePosition=list(pos), baseOrientation=list(orn), physicsClientId=self.cli)
        getattr(self.out, bucket).append(uid)
        return uid

    def box(self, centre: Sequence[float], half: Sequence[float], rgba: Sequence[float], *, yaw: float = 0.0,
            collide: bool = True, bucket: str = "props") -> int:
        """An axis box (turned by yaw) centred at `centre`."""
        return self._body(self._shape("box", tuple(half), rgba, collide), centre, quat_yaw(yaw), bucket)

    def boxes(self, parts: Sequence[Tuple[Vec3, Vec3]], rgba: Sequence[float], *, yaw: float = 0.0,
              origin: Vec3 = (0.0, 0.0, 0.0), bucket: str = "props") -> Optional[int]:
        """Many (centre, half_extents) boxes merged into ONE static body, centres relative to origin.

        A field of crop rows is a few hundred boxes; as separate bodies every platform probe
        would test each one, as one compound body it is a single test. None when parts is empty.
        """
        if not parts:
            return None
        p, cli = self.p, self.cli
        n = len(parts)
        centres = [list(c) for c, _ in parts]
        halves = [list(h) for _, h in parts]
        col = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX] * n, halfExtents=halves,
                                          collisionFramePositions=centres, physicsClientId=cli)
        vis = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX] * n, halfExtents=halves,
                                       visualFramePositions=centres, physicsClientId=cli)
        uid = self._body((vis, col), origin, quat_yaw(yaw), bucket)
        p.changeVisualShape(uid, -1, rgbaColor=list(rgba), specularColor=[0, 0, 0], physicsClientId=cli)
        return uid

    def cylinder(self, x: float, y: float, z0: float, radius: float, height: float, rgba: Sequence[float], *,
                 collide: bool = True, bucket: str = "props") -> int:
        """An upright cylinder standing on z0."""
        return self._body(self._shape("cylinder", (radius, height), rgba, collide), (x, y, z0 + height / 2.0),
                          bucket=bucket)

    def sphere(self, centre: Sequence[float], radius: float, rgba: Sequence[float], *, collide: bool = True,
               bucket: str = "props") -> int:
        """A sphere centred at `centre`."""
        return self._body(self._shape("sphere", (radius,), rgba, collide), centre, bucket=bucket)

    def rod(self, a: Vec3, b: Vec3, radius: float, rgba: Sequence[float], *, collision_radius: Optional[float] = None,
            collide: bool = True, bucket: str = "props") -> int:
        """A capsule from a to b; the collision radius may be thicker than the drawn one."""
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if length < 1e-6:
            raise ValueError("rod endpoints coincide")
        c_r = radius if collision_radius is None else collision_radius
        shapes = self._shape("capsule", (radius, length), rgba, collide, collision_dims=(c_r, length))
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)
        return self._body(shapes, mid, quat_z_to(d), bucket)

    def wire(self, span: WireSpan, rgba: Sequence[float], n_segments: int = 8) -> List[int]:
        """A sagging conductor as ONE compound body of thin capsules, registered for the ToF wire model.

        One body per span keeps a corridor of 30 spans at 30 bodies for the placement probe and
        gives the ToF model one segmentation id per conductor.
        """
        p, cli = self.p, self.cli
        pts = span.points(n_segments)
        origin = pts[0]
        c_r = max(span.radius_m, WIRE_COLLISION_RADIUS_MIN_M)
        frames, orns, lengths = [], [], []
        for a, b in zip(pts, pts[1:]):
            d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if length < 1e-6:
                continue
            frames.append([(a[0] + b[0]) / 2.0 - origin[0], (a[1] + b[1]) / 2.0 - origin[1],
                           (a[2] + b[2]) / 2.0 - origin[2]])
            orns.append(list(quat_z_to(d)))
            lengths.append(length)
        if not frames:
            raise ValueError("wire span has zero length")
        n = len(frames)
        col = p.createCollisionShapeArray(shapeTypes=[p.GEOM_CAPSULE] * n, radii=[c_r] * n, lengths=lengths,
                                          collisionFramePositions=frames, collisionFrameOrientations=orns,
                                          physicsClientId=cli)
        vis = p.createVisualShapeArray(shapeTypes=[p.GEOM_CAPSULE] * n, radii=[span.radius_m] * n, lengths=lengths,
                                       visualFramePositions=frames, visualFrameOrientations=orns,
                                       physicsClientId=cli)
        uid = self._body((vis, col), origin, bucket="beams")
        p.changeVisualShape(uid, -1, rgbaColor=list(rgba), specularColor=[0, 0, 0], physicsClientId=cli)
        self.out.wires.append(span)
        return [uid]

    # -- composites -----------------------------------------------------------
    def ground(self, rgba: Sequence[float], half_extent: float = MAP_HALF_EXTENT_M + 40.0,
               texture_path: Optional[str] = None) -> int:
        """The collidable floor slab with its top face at z = 0, like the forest floor."""
        uid = self.box((0.0, 0.0, -GROUND_HALF_THICKNESS_M), (half_extent, half_extent, GROUND_HALF_THICKNESS_M),
                       rgba, bucket="terrain")
        if texture_path:
            try:
                tex = self.p.loadTexture(texture_path, physicsClientId=self.cli)
                self.p.changeVisualShape(uid, -1, textureUniqueId=tex, physicsClientId=self.cli)
            except Exception:  # a missing texture only costs looks, never the map
                pass
        return uid

    def slab(self, x0: float, y0: float, x1: float, y1: float, top_z: float, thickness: float,
             rgba: Sequence[float], *, collide: bool = True, yaw: float = 0.0, bucket: str = "terrain") -> int:
        """A flat box between two corners whose top face sits at top_z."""
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = (abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0, thickness / 2.0)
        return self.box((cx, cy, top_z - thickness / 2.0), half, rgba, yaw=yaw, collide=collide, bucket=bucket)

    def tree(self, x: float, y: float, height: float, crown_r: float, trunk_rgba: Sequence[float],
             crown_rgba: Optional[Sequence[float]], *, crown: str = "round", z0: float = 0.0) -> List[int]:
        """A tree: a trunk plus a crown (round, cone, palm or none for a leaf-off tree)."""
        trunk_r = max(0.08, 0.03 * height)
        if crown == "none" or crown_rgba is None:
            uids = [self.cylinder(x, y, z0, trunk_r, height, trunk_rgba, bucket="trees")]
            # bare branches: two thin rods so a leaf-off crown still blocks something
            for k in (0, 1):
                ang = 1.3 + k * 2.4
                tip = (x + 0.45 * height * 0.35 * math.cos(ang), y + 0.45 * height * 0.35 * math.sin(ang), z0 + height)
                uids.append(self.rod((x, y, z0 + 0.6 * height), tip, trunk_r * 0.45, trunk_rgba, bucket="trees"))
            return uids
        if crown == "palm":
            uids = [self.cylinder(x, y, z0, trunk_r * 0.8, height, trunk_rgba, bucket="trees")]
            for k in range(4):
                ang = k * math.pi / 2.0 + 0.4
                tip = (x + crown_r * math.cos(ang), y + crown_r * math.sin(ang), z0 + height - 0.9)
                uids.append(self.rod((x, y, z0 + height), tip, 0.12, crown_rgba, bucket="trees"))
            return uids
        trunk_h = height * (0.45 if crown == "round" else 0.25)
        uids = [self.cylinder(x, y, z0, trunk_r, trunk_h + crown_r * 0.3, trunk_rgba, bucket="trees")]
        if crown == "cone":
            # a stack of two shrinking cylinders reads as a conifer without a mesh asset
            lower_h = (height - trunk_h) * 0.6
            uids.append(self.cylinder(x, y, z0 + trunk_h, crown_r, lower_h, crown_rgba, bucket="trees"))
            uids.append(self.cylinder(x, y, z0 + trunk_h + lower_h, crown_r * 0.55, height - trunk_h - lower_h,
                                      crown_rgba, bucket="trees"))
        else:
            uids.append(self.sphere((x, y, z0 + height - crown_r), crown_r, crown_rgba, bucket="trees"))
        return uids

    def hut(self, x: float, y: float, w: float, d: float, h: float, wall_rgba: Sequence[float],
            roof_rgba: Sequence[float], *, yaw: float = 0.0, z0: float = 0.0) -> List[int]:
        """A one-room hut: walls plus a slightly wider flat roof slab."""
        walls = self.box((x, y, z0 + h / 2.0), (w / 2.0, d / 2.0, h / 2.0), wall_rgba, yaw=yaw)
        roof = self.box((x, y, z0 + h + 0.08), (w / 2.0 + 0.25, d / 2.0 + 0.25, 0.08), roof_rgba, yaw=yaw)
        return [walls, roof]

    def summary(self, **info: object) -> Dict[str, object]:
        """Finish: store the counts and any builder facts, and hand back the asset dict."""
        out = self.out
        out.info.update(info)
        out.info.update({
            "n_trees": len(out.trees), "n_props": len(out.props), "n_beams": len(out.beams),
            "n_terrain": len(out.terrain), "n_wires": len(out.wires),
            "n_bodies": len(out.trees) + len(out.props) + len(out.beams) + len(out.terrain),
        })
        return out.as_assets()


__all__ = [
    "BuiltMap", "CLEARING_MIN_M", "GROUND_HALF_THICKNESS_M", "Kit", "MAP_HALF_EXTENT_M",
    "WIRE_COLLISION_RADIUS_MIN_M", "WireSpan", "biome_cache_dir", "catenary_points", "chord_hole",
    "is_clear", "quat_yaw", "quat_z_to", "segment_clear", "subtract_intervals", "tint",
]
