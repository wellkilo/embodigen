"""Robot episodes -> 4D worlds, sampled from the scene's exact geometry.

A packed episode keeps the scene (every box and sphere with its size, place
and albedo) and, per frame, where the moved objects, the end effector and
the gripper are. That is the whole world, so splats are sampled straight
from those surfaces: complete from every side, not the single-view shell a
depth map would give, and with no model in the loop. Objects that never move
are static; the moved ones and the gripper fingers are sampled once in their
own frame and placed per frame (rigid motion, as in the simulation).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from embodigen.geometry import normalize
from embodigen.modules.export import _gripper_fingers
from embodigen.modules.pack import read_episode
from embodigen.types import ObjectSpec
from embodigen.world4d.gaussians import Gaussians
from embodigen.world4d.world import View, World

LIGHT = normalize((0.35, -0.45, 0.82))
BIG_AREA = 1.0  # m^2: surfaces this large (the table) are cropped to the workspace
MARGIN = 0.35  # m of table kept around the objects


def _shade(albedo: tuple[int, int, int], normal: tuple[float, float, float]) -> tuple[float, float, float]:
    k = 0.58 + 0.42 * max(0.0, sum(a * b for a, b in zip(normal, LIGHT)))
    return tuple(min(1.0, c / 255.0 * k) for c in albedo)  # type: ignore[return-value]


def sample_object(obj: ObjectSpec, spacing: float, crop: tuple[list[float], list[float]] | None = None,
                  origin: tuple[float, float, float] | None = None) -> Gaussians:
    """Surfels on the object's surface (bottom faces skipped: nothing looks from below).
    ``origin`` shifts positions (sample in an object's own frame); ``crop`` limits x/y."""
    g = Gaussians()
    cx, cy, cz = origin if origin is not None else obj.center
    if obj.kind == "sphere":
        r = obj.half_extent[0]
        n = max(12, math.ceil(4 * math.pi * r * r / spacing ** 2))
        golden = math.pi * (3 - math.sqrt(5))
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            rad = math.sqrt(max(0.0, 1 - z * z))
            nrm = (math.cos(golden * i) * rad, math.sin(golden * i) * rad, z)
            g.surfel((cx + r * nrm[0], cy + r * nrm[1], cz + r * nrm[2]), nrm, spacing * 0.7, _shade(obj.albedo, nrm))
        return g
    half = obj.half_extent
    for axis in range(3):
        for sign in (1.0, -1.0):
            if axis == 2 and sign < 0:
                continue
            nrm = tuple(sign if k == axis else 0.0 for k in range(3))
            u_ax, v_ax = [k for k in range(3) if k != axis]
            lo = [-half[u_ax], -half[v_ax]]
            hi = [half[u_ax], half[v_ax]]
            if crop is not None and axis < 2 and not crop[0][axis] <= (cx, cy)[axis] + sign * half[axis] <= crop[1][axis]:
                continue  # the face itself lies outside the window
            if crop is not None:  # world-space x/y window, in the face's axes
                for j, ax in enumerate((u_ax, v_ax)):
                    if ax < 2:
                        c = (cx, cy)[ax]
                        lo[j] = max(lo[j], crop[0][ax] - c)
                        hi[j] = min(hi[j], crop[1][ax] - c)
                if lo[0] >= hi[0] or lo[1] >= hi[1]:
                    continue
            nu = max(1, math.ceil((hi[0] - lo[0]) / spacing))
            nv = max(1, math.ceil((hi[1] - lo[1]) / spacing))
            du, dv = (hi[0] - lo[0]) / nu, (hi[1] - lo[1]) / nv
            color = _shade(obj.albedo, nrm)  # type: ignore[arg-type]
            radius = 0.7 * max(du, dv)
            for a in range(nu):
                for b in range(nv):
                    p = [0.0, 0.0, 0.0]
                    p[axis] = sign * half[axis]
                    p[u_ax] = lo[0] + (a + 0.5) * du
                    p[v_ax] = lo[1] + (b + 0.5) * dv
                    g.surfel((cx + p[0], cy + p[1], cz + p[2]), nrm, radius, color)  # type: ignore[arg-type]
    return g


def _moved(g: Gaussians, offset: tuple[float, float, float]) -> Gaussians:
    out = Gaussians()
    out.extend(g)
    for i in range(len(out)):
        for k in range(3):
            out.pos[3 * i + k] += offset[k]
    return out


def episode_world(episode: Path, spacing: float = 0.006) -> World:
    episode = Path(episode)
    scene, frames, _ = read_episode(episode)
    moved = {name for f in frames for name in f.objects}
    small = [o for o in scene.objects if 4 * o.half_extent[0] * o.half_extent[1] < BIG_AREA]
    xs = [o.center[0] + s * o.half_extent[0] for o in small for s in (-1, 1)] + [f.eef[0] for f in frames]
    ys = [o.center[1] + s * o.half_extent[1] for o in small for s in (-1, 1)] + [f.eef[1] for f in frames]
    crop = ([min(xs) - MARGIN, min(ys) - MARGIN], [max(xs) + MARGIN, max(ys) + MARGIN])

    static = Gaussians()
    for obj in scene.objects:
        if obj.name in moved:
            continue
        big = 4 * obj.half_extent[0] * obj.half_extent[1] >= BIG_AREA
        static.extend(sample_object(obj, spacing * (2 if big else 1), crop if big else None))

    local = {o.name: sample_object(o, spacing, origin=(0.0, 0.0, 0.0)) for o in scene.objects if o.name in moved}
    fingers = {f.name: sample_object(f, spacing * 0.6, origin=(0.0, 0.0, 0.0))
               for f in _gripper_fingers(frames[0].eef, frames[0].gripper)}
    dynamic, views = [], []
    cam = scene.camera
    view = View(eye=tuple(cam.eye), target=tuple(cam.target), up=tuple(cam.up),
                fov_y=math.degrees(2 * math.atan(cam.height / 2 / cam.fy)))
    for f in frames:
        dyn = Gaussians()
        for obj in scene.objects:
            if obj.name in local:
                dyn.extend(_moved(local[obj.name], f.objects.get(obj.name, obj.center)))
        for finger in _gripper_fingers(f.eef, f.gripper):
            dyn.extend(_moved(fingers[finger.name], finger.center))
        dynamic.append(dyn)
        views.append(view)
    info = json.loads((episode / "meta/info.json").read_text(encoding="utf-8"))
    fidelity = json.loads((episode / "meta/fidelity.json").read_text(encoding="utf-8"))
    return World(
        name=episode.name,
        fps=float(info.get("fps", 8.0)),
        static=static,
        dynamic=dynamic,
        views=views,
        source={"kind": "robot-episode", "episode": str(episode), "instruction": scene.instruction,
                "robot": scene.robot, "template": scene.template, "fidelity_p95_mm": fidelity.get("p95_error_mm"),
                "passed": fidelity.get("passed"), "spacing_m": spacing,
                "method": "surfels sampled from the scene's exact boxes and spheres; moved objects and gripper per frame"},
        layers={"static": "table (cropped to the workspace) and fixed objects",
                "dynamic": "moved objects and gripper fingers, per frame"},
    )
