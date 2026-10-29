"""3D Gaussians: a columnar set, and the two file formats viewers read.

* ``.ply``: the 3D Gaussian Splatting layout of Kerbl et al. (2023) that
  SuperSplat, the INRIA viewer and most tools load: per splat ``x y z``,
  ``nx ny nz``, ``f_dc_0..2`` (degree-0 spherical harmonics), ``opacity``
  (logit), ``scale_0..2`` (log) and ``rot_0..3`` (quaternion w, x, y, z).
* ``.splat``: the compact 32-byte record used by web splat viewers
  (antimatter15/splat): position and linear scale as float32, RGBA as uint8,
  and the quaternion as uint8 (w, x, y, z mapped from [-1, 1]).

Splats here are surfels: flat discs on a surface, thin along its normal, so
a set sampled from exact geometry renders as that surface from any side.
"""

from __future__ import annotations

import math
import struct
import sys
from array import array
from pathlib import Path

SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic
THIN = 0.12  # surfel thickness relative to its radius

Vec3 = tuple[float, float, float]


def quat_from_z(n: Vec3) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) rotating +z onto the unit vector n."""
    nx, ny, nz = n
    if nz < -0.999999:  # opposite: half turn about x
        return (0.0, 1.0, 0.0, 0.0)
    w, x, y, z = 1.0 + nz, -ny, nx, 0.0  # (1 + z.n, z x n)
    s = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / s, x / s, y / s, z / s)


class Gaussians:
    """Parallel float arrays, one entry per splat; colours are linear 0..1 RGB."""

    __slots__ = ("pos", "scale", "rot", "rgb", "alpha")

    def __init__(self) -> None:
        self.pos = array("f")
        self.scale = array("f")
        self.rot = array("f")
        self.rgb = array("f")
        self.alpha = array("f")

    def __len__(self) -> int:
        return len(self.alpha)

    def add(self, p: Vec3, scale: Vec3, rot: tuple[float, float, float, float], rgb: Vec3,
            alpha: float = 0.95) -> None:
        self.pos.extend(p)
        self.scale.extend(scale)
        self.rot.extend(rot)
        self.rgb.extend(rgb)
        self.alpha.append(alpha)

    def surfel(self, p: Vec3, normal: Vec3, radius: float, rgb: Vec3, alpha: float = 0.95) -> None:
        self.add(p, (radius, radius, radius * THIN), quat_from_z(normal), rgb, alpha)

    def extend(self, other: "Gaussians") -> None:
        for name in self.__slots__:
            getattr(self, name).extend(getattr(other, name))

    def bounds(self) -> tuple[list[float], list[float]]:
        if not len(self):
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lo = [min(self.pos[i::3]) for i in range(3)]
        hi = [max(self.pos[i::3]) for i in range(3)]
        return lo, hi

    # -- files -------------------------------------------------------------

    def write_ply(self, path: Path) -> Path:
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "comment embodigen world4d: surfels from exact geometry\n"
                  f"element vertex {len(self)}\n" + "".join(f"property float {n}\n" for n in names)
                  + "end_header\n")
        out = array("f")
        for i in range(len(self)):
            a = min(max(self.alpha[i], 1e-4), 1 - 1e-4)
            out.extend(self.pos[3 * i : 3 * i + 3])
            out.extend((0.0, 0.0, 0.0))
            out.extend((c - 0.5) / SH_C0 for c in self.rgb[3 * i : 3 * i + 3])
            out.append(math.log(a / (1 - a)))
            out.extend(math.log(max(s, 1e-7)) for s in self.scale[3 * i : 3 * i + 3])
            out.extend(self.rot[4 * i : 4 * i + 4])
        if sys.byteorder == "big":
            out.byteswap()  # the file is little-endian whatever the host
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header.encode("ascii") + out.tobytes())
        return path

    def splat_bytes(self) -> bytes:
        rec = struct.Struct("<6f4B4B")
        chunks = []
        for i in range(len(self)):
            r, g, b = (min(255, max(0, round(c * 255))) for c in self.rgb[3 * i : 3 * i + 3])
            q = (min(255, max(0, round(v * 128 + 128))) for v in self.rot[4 * i : 4 * i + 4])
            chunks.append(rec.pack(*self.pos[3 * i : 3 * i + 3], *self.scale[3 * i : 3 * i + 3],
                                   r, g, b, min(255, max(0, round(self.alpha[i] * 255))), *q))
        return b"".join(chunks)

    def write_splat(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.splat_bytes())
        return path


def read_ply(path: Path) -> dict[str, list[float]]:
    """Columns of a binary little-endian float PLY (enough to check what we wrote)."""
    blob = Path(path).read_bytes()
    end = blob.index(b"end_header\n") + len(b"end_header\n")
    names, count = [], 0
    for line in blob[:end].decode("ascii").splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts[:2] == ["property", "float"]:
            names.append(parts[2])
    values = struct.unpack(f"<{count * len(names)}f", blob[end : end + 4 * count * len(names)])
    return {n: list(values[i :: len(names)]) for i, n in enumerate(names)}
