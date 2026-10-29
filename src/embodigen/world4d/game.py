"""Engine game recordings -> 4D worlds, lifted from exact depth and pose.

This is the 3D4D route (frames -> depth -> points per frame -> splats) with
the guesses removed: the depth is the engine's own, the camera is the
player's recorded position, yaw and pitch, and the labels say which pixel is
wall, floor or an entity. Walls and floor are static: lifted from every
frame, merged on a voxel grid (keeping the colour seen from closest), stored
once. Entities move, so their pixels stay with their frame.

The camera model is the raycast engine's (``gameplay/envs/raycast.py``): a
column ``c`` looks along ``forward + right * plane * (2c/w - 1)``, depth is
the distance along ``forward``, the eye is 0.5 above the floor and a wall of
height 1 at depth ``d`` spans ``h / d`` rows centred on the horizon.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from embodigen.gameplay.recording import read_meta
from embodigen.gameplay.registry import WorldRegistry
from embodigen.pngio import read_png
from embodigen.world4d.gaussians import Gaussians
from embodigen.world4d.world import View, World

EYE = 0.5
CEILING = 0.98  # label-0 points this high are the ceiling (left out by default, so the level can be seen from above)


def _obs_with_truth(raw: Path) -> list[dict]:
    rows = []
    with (raw / "obs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line) if line.strip() else None
            if row and row.get("depth") and row.get("labels") and row.get("state") and "pos" in row["state"]:
                rows.append(row)
    return rows


def recording_world(raw: Path, voxel: float = 0.06, stride: int = 1, ceiling: bool = False,
                    registry: WorldRegistry | None = None) -> World:
    raw = Path(raw)
    meta = read_meta(raw)
    if not meta.engine.startswith("mock-raycast"):
        raise SystemExit(f"{raw}: 4D worlds are lifted from raycast-engine rollouts (exact depth and pose); "
                         f"this recording comes from {meta.source} / {meta.engine or 'a dataset'}")
    rows = _obs_with_truth(raw)
    if not rows:
        raise SystemExit(f"{raw}: no frames with engine depth, labels and pose (recorded with --no-gt?)")
    spec = (registry or WorldRegistry.load()).get(meta.world)
    plane = math.tan(math.radians(float(spec.engine_args.get("fov_deg", 66.0))) / 2)
    w, h = meta.width, meta.height

    static: dict[tuple[int, int, int], tuple[float, tuple, tuple, float]] = {}  # voxel -> (dist, pos, normal, rgb)
    dynamic, views = [], []
    for row in rows:
        st = row["state"]
        px, py = st["pos"][:2]
        yaw = math.radians(st["yaw"])
        pitch = math.radians(st.get("pitch", 0.0))  # the engine's horizon shift, stored through degrees()
        fx, fy, rx, ry = math.cos(yaw), math.sin(yaw), math.sin(yaw), -math.cos(yaw)
        horizon = h / 2 - pitch * h / 2
        iw, ih, ch, rgb = read_png(raw / row["frame"])
        depth = memoryview((raw / row["depth"]).read_bytes()).cast("f")
        labels = memoryview((raw / row["labels"]).read_bytes()).cast("H")
        dyn = Gaussians()
        for r in range(0, h, stride):
            for col in range(0, w, stride):
                k = r * w + col
                d = depth[k]
                if not 0 < d < 1e4:
                    continue
                cam = 2.0 * col / w - 1.0
                ray_x, ray_y = fx + rx * plane * cam, fy + ry * plane * cam
                x, y = px + d * ray_x, py + d * ray_y
                z = EYE + (horizon - r) * d / h
                color = tuple(c / 255.0 for c in rgb[k * ch : k * ch + 3])
                lab = labels[k]
                foot = d * max(2 * plane / w, 1.0 / h) * stride  # one pixel's footprint at this depth
                if lab >= 2:  # an entity: a billboard facing the player
                    n = math.hypot(ray_x, ray_y)
                    dyn.surfel((x, y, z), (-ray_x / n, -ray_y / n, 0.0), max(foot * 0.75, 0.01), color)
                    continue
                if lab == 0:
                    if z > CEILING * 2 * EYE and not ceiling:
                        continue
                    normal = (0.0, 0.0, -1.0) if z > EYE else (0.0, 0.0, 1.0)
                else:  # a wall: axis-aligned, facing the player
                    ex, ey = abs(x - round(x)), abs(y - round(y))
                    normal = (-math.copysign(1.0, ray_x), 0.0, 0.0) if ex < ey else (0.0, -math.copysign(1.0, ray_y), 0.0)
                key = (round(x / voxel), round(y / voxel), round(z / voxel))
                seen = static.get(key)
                if seen is None or d < seen[0]:
                    static[key] = (d, (x, y, z), normal, color)
        dynamic.append(dyn)
        views.append(View(eye=(px, py, EYE), target=(px + fx, py + fy, EYE + pitch / 2),
                          fov_y=math.degrees(2 * math.atan(0.5))))
    g = Gaussians()
    for d, p, n, c in static.values():
        g.surfel(p, n, voxel * 0.75, c)
    return World(
        name=meta.recording_id,
        fps=5.0,
        static=g,
        dynamic=dynamic,
        views=views,
        source={"kind": "game-recording", "recording": str(raw), "world": meta.world, "title": spec.title,
                "seed": meta.seed, "frames_lifted": len(rows), "voxel_m": voxel, "stride": stride,
                "ceiling": ceiling,
                "method": "pixels lifted with the engine's depth and the recorded pose; walls/floor merged on a "
                          "voxel grid (static), entities per frame (dynamic)"},
        layers={"static": "walls and floor seen during the recording" + (" and ceiling" if ceiling else ""),
                "dynamic": "enemies, pickups and exits as the player saw them, per frame"},
    )
