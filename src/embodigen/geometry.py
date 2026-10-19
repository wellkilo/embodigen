"""Small vector / camera utilities.

Kept dependency-free so the contract can be exercised on a laptop with no
GPU. The camera model produces the same per-pixel ray representation the
photorealizer needs downstream (Pluecker-style: direction + moment), which is
what keeps the camera trajectory locked during rendering.
"""

from __future__ import annotations

import math

from embodigen.types import Camera, Vec3

EPS = 1e-9


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    n = norm(a)
    if n < EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def dist(a: Vec3, b: Vec3) -> float:
    return norm(sub(a, b))


def lerp(a: Vec3, b: Vec3, s: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    )


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def ray_box(origin: Vec3, direction: Vec3, center: Vec3, half: Vec3) -> float | None:
    """Slab test against an axis-aligned box. Returns entry distance or None."""
    t_near = -math.inf
    t_far = math.inf
    for i in range(3):
        o = origin[i] - center[i]
        d = direction[i]
        lo = -half[i]
        hi = half[i]
        if abs(d) < EPS:
            if o < lo or o > hi:
                return None
            continue
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return None
    if t_far < 0:
        return None
    return t_near if t_near > 0 else t_far


def ray_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: float) -> float | None:
    oc = sub(origin, center)
    b = dot(oc, direction)
    c = dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc
    if t0 > 0:
        return t0
    if t1 > 0:
        return t1
    return None


class CameraModel:
    """Pinhole camera with a fixed basis, precomputed per-pixel rays."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self.forward = normalize(sub(camera.target, camera.eye))
        right = cross(self.forward, normalize(camera.up))
        self.right = normalize(right)
        self.up = cross(self.right, self.forward)
        self._rays: list[tuple[Vec3, float]] | None = None

    def ray(self, u: int, v: int) -> tuple[Vec3, float]:
        """Return (normalised direction, cos theta) for pixel (u, v)."""
        c = self.camera
        x = (u + 0.5 - c.cx) / c.fx
        y = (v + 0.5 - c.cy) / c.fy
        d = normalize(add(add(self.forward, mul(self.right, x)), mul(self.up, -y)))
        return d, dot(d, self.forward)

    @property
    def rays(self) -> list[tuple[Vec3, float]]:
        if self._rays is None:
            self._rays = [
                self.ray(u, v)
                for v in range(self.camera.height)
                for u in range(self.camera.width)
            ]
        return self._rays

    def unproject(self, u: float, v: float, planar_z: float) -> Vec3:
        """Back-project a pixel with a planar depth into world space."""
        d, cos_z = self.ray(int(min(max(u, 0), self.camera.width - 1)),
                            int(min(max(v, 0), self.camera.height - 1)))
        if abs(cos_z) < EPS:
            return self.camera.eye
        return add(self.camera.eye, mul(d, planar_z / cos_z))

    def project(self, point: Vec3) -> tuple[float, float, float]:
        """Project a world point to (u, v, planar_z).

        Uses the same pixel-centre convention as :meth:`ray`, i.e. pixel (u, v)
        covers [u, u+1) with its centre at u + 0.5.
        """
        c = self.camera
        rel = sub(point, c.eye)
        z = dot(rel, self.forward)
        if abs(z) < EPS:
            return (c.cx, c.cy, 0.0)
        x = dot(rel, self.right) / z
        y = dot(rel, self.up) / z
        return (c.cx + x * c.fx - 0.5, c.cy - y * c.fy - 0.5, z)

    def plucker(self) -> list[tuple[float, float, float, float, float, float]]:
        """Per-pixel Pluecker embedding (direction, moment).

        This is the form a video model wants for camera control: it encodes
        the full ray, not just the camera pose.
        """
        out = []
        for d, _ in self.rays:
            m = cross(self.camera.eye, d)
            out.append((d[0], d[1], d[2], m[0], m[1], m[2]))
        return out

    def extrinsics(self) -> list[list[float]]:
        """World-to-camera matrix as a 4x4 row-major list."""
        r, u, f = self.right, self.up, self.forward
        eye = self.camera.eye
        return [
            [r[0], r[1], r[2], -dot(r, eye)],
            [-u[0], -u[1], -u[2], dot(u, eye)],
            [f[0], f[1], f[2], -dot(f, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def intrinsics(self) -> list[float]:
        c = self.camera
        return [c.fx, c.fy, c.cx, c.cy]
