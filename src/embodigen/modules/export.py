"""GTExporter: (scene, trajectory) -> per-frame ground truth.

This is a small software rasteriser/ray caster standing in for ManiSkill3. It
ray-casts the same primitives the engine would simulate, so depth,
segmentation, object poses and the end-effector trajectory are all exact by
construction rather than estimated. Swapping in ManiSkill3 means implementing
``export`` against SAPIEN's renderer and state vectors; the contract does not
change.
"""

from __future__ import annotations

import math

from embodigen.geometry import CameraModel, add, normalize, ray_box, ray_sphere, sub
from embodigen.types import FrameGT, ObjectSpec, SceneSpec, Trajectory, Vec3

SKY_TOP = (126, 138, 152)
SKY_BOTTOM = (58, 62, 70)
LIGHT_DIR = normalize((-0.45, -0.55, 0.78))


def _gripper_fingers(eef: Vec3, gripper: float) -> list[ObjectSpec]:
    sep = gripper / 2.0 + 0.008
    half_h = 0.018
    z = eef[2] + half_h
    return [
        ObjectSpec(
            name="finger_l",
            kind="box",
            center=(eef[0], eef[1] - sep, z),
            half_extent=(0.012, 0.008, half_h),
            material="gripper",
            albedo=(150, 152, 156),
            movable=False,
        ),
        ObjectSpec(
            name="finger_r",
            kind="box",
            center=(eef[0], eef[1] + sep, z),
            half_extent=(0.012, 0.008, half_h),
            material="gripper",
            albedo=(150, 152, 156),
            movable=False,
        ),
    ]


def _clay(albedo: tuple[int, int, int]) -> Vec3:
    """White-model look: mostly neutral clay with a hint of the true albedo."""
    lum = (0.299 * albedo[0] + 0.587 * albedo[1] + 0.114 * albedo[2]) / 255.0
    base = 0.55 + 0.35 * lum
    return (
        min(1.0, base * 0.96 + albedo[0] / 255.0 * 0.06),
        min(1.0, base * 0.96 + albedo[1] / 255.0 * 0.06),
        min(1.0, base * 0.94 + albedo[2] / 255.0 * 0.08),
    )


def _shade(albedo: Vec3, normal: Vec3, ambient: float = 0.32) -> tuple[int, int, int]:
    lambert = max(0.0, normal[0] * LIGHT_DIR[0] + normal[1] * LIGHT_DIR[1] + normal[2] * LIGHT_DIR[2])
    k = ambient + (1.0 - ambient) * lambert
    return (
        int(min(255, albedo[0] * k * 255)),
        int(min(255, albedo[1] * k * 255)),
        int(min(255, albedo[2] * k * 255)),
    )


def _normal_at(obj: ObjectSpec, point: Vec3) -> Vec3:
    if obj.kind == "sphere":
        return normalize(sub(point, obj.center))
    local = [
        (point[i] - obj.center[i]) / max(obj.half_extent[i], 1e-9) for i in range(3)
    ]
    axis = max(range(3), key=lambda i: abs(local[i]))
    n = [0.0, 0.0, 0.0]
    n[axis] = 1.0 if local[axis] >= 0 else -1.0
    return (n[0], n[1], n[2])


class SoftwareGTExporter:
    name = "software-raycast-v0"

    def export(self, scene: SceneSpec, trajectory: Trajectory) -> list[FrameGT]:
        cam = CameraModel(scene.camera)
        rays = cam.rays
        frames: list[FrameGT] = []

        for frame in trajectory.frames:
            instances: list[ObjectSpec] = []
            for obj in scene.objects:
                if obj.name in frame.objects:
                    obj = obj.moved_to(frame.objects[obj.name])
                instances.append(obj)
            instances.extend(_gripper_fingers(frame.eef, frame.gripper))
            labels = {obj.name: i + 1 for i, obj in enumerate(instances)}

            rgb, depth, seg = self._render(cam, rays, instances)
            frames.append(
                FrameGT(
                    t=frame.t,
                    rgb=bytes(rgb),
                    depth=depth,
                    seg=seg,
                    eef=frame.eef,
                    gripper=frame.gripper,
                    objects=dict(frame.objects),
                    action=frame.action,
                    seg_labels=labels,
                )
            )
        return frames

    def _render(
        self, cam: CameraModel, rays: list[tuple[Vec3, float]], instances: list[ObjectSpec]
    ) -> tuple[bytearray, list[float], list[int]]:
        c = cam.camera
        n_px = c.width * c.height
        far = c.far
        origin = c.eye
        rgb = bytearray(n_px * 3)
        depth = [far] * n_px
        seg = [0] * n_px

        boxes = [
            (i, inst.center, inst.half_extent)
            for i, inst in enumerate(instances)
            if inst.kind == "box"
        ]
        spheres = [
            (i, inst.center, inst.half_extent[0])
            for i, inst in enumerate(instances)
            if inst.kind == "sphere"
        ]

        for p, (direction, cos_z) in enumerate(rays):
            best_t = far
            best_i = -1

            for i, center, half in boxes:
                t = ray_box(origin, direction, center, half)
                if t is not None and 0.0 < t < best_t:
                    best_t = t
                    best_i = i
            for i, center, radius in spheres:
                t = ray_sphere(origin, direction, center, radius)
                if t is not None and 0.0 < t < best_t:
                    best_t = t
                    best_i = i

            if best_i < 0:
                k = max(0.0, min(1.0, (p // c.width) / max(1, c.height - 1)))
                rgb[p * 3] = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * k)
                rgb[p * 3 + 1] = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * k)
                rgb[p * 3 + 2] = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * k)
                continue

            obj = instances[best_i]
            point = add(origin, (direction[0] * best_t, direction[1] * best_t, direction[2] * best_t))
            color = _shade(_clay(obj.albedo), _normal_at(obj, point))
            rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2] = color
            depth[p] = best_t * cos_z
            seg[p] = best_i + 1

        return rgb, depth, seg


def angular_size(radius: float, distance: float) -> float:
    """Handy for sanity-checking that a marker spans enough pixels."""
    return 2.0 * math.atan(radius / max(distance, 1e-9))
