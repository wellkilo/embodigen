"""A 4D world on disk: static splats, per-frame dynamic splats, cameras.

    <world>/
    ├── world.json             what a viewer needs: fps, frames, cameras, files, bounds
    ├── static.splat           splats present in every frame (table, walls, floor)
    ├── dynamic/000000.splat   splats of frame i only (moved objects, gripper, enemies)
    └── ply/frame_000000.ply   static + dynamic of frame i, 3DGS layout (SuperSplat etc.)

Splitting static from dynamic is the 4D decomposition: geometry that never
moves is stored once, so a long sequence costs little more than its first
frame. It comes from exact labels (which object moved, which pixel is an
enemy), not from a model guessing it.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodigen.world4d.gaussians import Gaussians

FORMAT = "embodigen-4dgs-v1"
PLY_BYTES = 17 * 4  # one splat in the PLY layout
PLY_BUDGET = 64 * 1024 * 1024


@dataclass
class View:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_y: float = 50.0  # degrees

    def to_json(self) -> dict[str, Any]:
        r = lambda v: [round(x, 5) for x in v]  # noqa: E731
        return {"eye": r(self.eye), "target": r(self.target), "up": r(self.up), "fov_y": round(self.fov_y, 3)}


@dataclass
class World:
    name: str
    fps: float
    static: Gaussians
    dynamic: list[Gaussians]  # one per frame (may be empty)
    views: list[View]  # one per frame: the recording's own camera
    source: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, str] = field(default_factory=dict)  # what static / dynamic hold, for the page

    @property
    def frames(self) -> int:
        return len(self.dynamic)


def write_world(world: World, out: Path, ply: str = "all") -> dict[str, Any]:
    """Write the world; ``ply`` is auto | all | first | none (the PLY sequence is for other tools;
    auto writes every frame while that stays under ``PLY_BUDGET`` bytes, else the first)."""
    out = Path(out)
    if out.exists():
        if not (out / "world.json").exists():
            raise SystemExit(f"{out} exists and is not a world; refusing to overwrite it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    world.static.write_splat(out / "static.splat")
    files = []
    lo, hi = world.static.bounds()
    for i, dyn in enumerate(world.dynamic):
        name = f"dynamic/{i:06d}.splat" if len(dyn) else ""
        if name:
            dyn.write_splat(out / name)
            dlo, dhi = dyn.bounds()
            lo = [min(a, b) for a, b in zip(lo, dlo)] if len(world.static) else dlo
            hi = [max(a, b) for a, b in zip(hi, dhi)] if len(world.static) else dhi
        files.append({"dynamic": name, "count": len(dyn)})
    if ply == "auto":
        total = sum(len(world.static) + len(d) for d in world.dynamic) * PLY_BYTES
        ply = "all" if total <= PLY_BUDGET else "first"
    plys = range(world.frames) if ply == "all" else range(min(1, world.frames)) if ply == "first" else range(0)
    for i in plys:
        full = Gaussians()
        full.extend(world.static)
        full.extend(world.dynamic[i])
        full.write_ply(out / "ply" / f"frame_{i:06d}.ply")
        files[i]["ply"] = f"ply/frame_{i:06d}.ply"
    center = [(a + b) / 2 for a, b in zip(lo, hi)]
    radius = max(1e-3, math.dist(lo, hi) / 2)
    meta = {
        "format": FORMAT,
        "name": world.name,
        "fps": world.fps,
        "frames": world.frames,
        "up": [0.0, 0.0, 1.0],
        "static": {"file": "static.splat", "count": len(world.static)},
        "per_frame": files,
        "cameras": [v.to_json() for v in world.views],
        "bounds": {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi],
                   "center": [round(v, 4) for v in center], "radius": round(radius, 4)},
        "layers": world.layers,
        "splat_record": "32 bytes: float32 x,y,z, float32 scale x,y,z, uint8 r,g,b,a, uint8 quat w,x,y,z (v*128+128)",
        "source": world.source,
    }
    (out / "world.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"out": str(out), "frames": world.frames, "static": len(world.static),
            "dynamic_max": max((len(d) for d in world.dynamic), default=0),
            "ply": len(plys), "bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file())}
