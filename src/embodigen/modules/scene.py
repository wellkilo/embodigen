"""SceneGenerator: instruction -> scene spec.

This implementation is the "template + parameterisation" baseline described in
the design doc: a small set of templates, randomised parameters, and a
feasibility check that resamples until the task is physically achievable.
A code agent can later replace the template choice with free-form generation
against the same output contract.
"""

from __future__ import annotations

import random
from dataclasses import replace

from embodigen.geometry import dist
from embodigen.types import Camera, ObjectSpec, SceneSpec, TaskSpec, Vec3

COLOR_WORDS: dict[str, tuple[int, int, int]] = {
    "red": (198, 62, 58),
    "红": (198, 62, 58),
    "green": (86, 148, 96),
    "绿": (86, 148, 96),
    "blue": (66, 102, 176),
    "蓝": (66, 102, 176),
    "yellow": (208, 172, 74),
    "黄": (208, 172, 74),
    "orange": (204, 124, 66),
    "橙": (204, 124, 66),
    "purple": (134, 92, 166),
    "紫": (134, 92, 166),
    "black": (58, 58, 62),
    "黑": (58, 58, 62),
    "white": (206, 206, 202),
    "白": (206, 206, 202),
}


def _extract_color(instruction: str) -> tuple[int, int, int] | None:
    text = instruction.lower()
    for word, rgb in COLOR_WORDS.items():
        if word in text:
            return rgb
    return None


class TemplateSceneGenerator:
    """Sample a pick-and-place scene from a parameterised template."""

    name = "template-scene-v0"

    def __init__(self, camera: Camera | None = None, max_resamples: int = 32) -> None:
        self.camera = camera or Camera()
        self.max_resamples = max_resamples

    def generate(self, task: TaskSpec, robot: dict, rng: random.Random) -> SceneSpec:
        template = task.template or self._infer_template(task.instruction)
        if template == "pick_place":
            return self._pick_place(task, robot, rng)
        raise SystemExit(f"no template for instruction: {task.instruction!r}")

    @staticmethod
    def _infer_template(instruction: str) -> str:
        text = instruction.lower()
        if any(k in text for k in ("放进", "放入", "放到", "put", "place", "into", "in the")):
            return "pick_place"
        return "pick_place"

    def _pick_place(self, task: TaskSpec, robot: dict, rng: random.Random) -> SceneSpec:
        reach = float(robot.get("reach", 0.85))
        base: Vec3 = tuple(robot.get("base", (0.0, 0.0, 0.0)))  # type: ignore[assignment]
        albedo = _extract_color(task.instruction) or (198, 62, 58)
        notes: list[str] = []
        resamples = 0

        while True:
            cube_xy = (rng.uniform(-0.16, -0.02), rng.uniform(0.02, 0.16))
            bin_xy = (rng.uniform(0.10, 0.20), rng.uniform(-0.14, -0.02))
            cube_center: Vec3 = (cube_xy[0], cube_xy[1], 0.02)
            bin_center: Vec3 = (bin_xy[0], bin_xy[1], 0.0)

            ok = (
                dist(cube_center, base) < reach * 0.55
                and dist(bin_center, base) < reach * 0.6
                and dist(cube_center, bin_center) > 0.16
            )
            if ok:
                break
            resamples += 1
            if resamples >= self.max_resamples:
                notes.append("fell back to a fixed layout after resampling")
                cube_center = (-0.07, 0.08, 0.02)
                bin_center = (0.14, -0.06, 0.0)
                break
        if resamples:
            notes.append(f"resampled {resamples}x to satisfy reach/clearance")

        wall = 0.008
        half = 0.055
        height = 0.03
        bx, by, _ = bin_center
        objects = [
            ObjectSpec(
                name="table",
                kind="box",
                center=(0.0, 0.0, -0.02),
                half_extent=(0.30, 0.30, 0.02),
                material="table",
                albedo=(196, 190, 178),
                movable=False,
            ),
            ObjectSpec(
                name="cube",
                kind="box",
                center=cube_center,
                half_extent=(0.02, 0.02, 0.02),
                material="cube",
                albedo=albedo,
            ),
            # Hollow bin: floor plus four walls, so the opening reads in the render.
            ObjectSpec(
                name="bin_floor",
                kind="box",
                center=(bx, by, height / 2),
                half_extent=(half, half, wall),
                material="bin",
                albedo=(92, 96, 104),
                movable=False,
                is_container=True,
            ),
            ObjectSpec(
                name="bin_wall_x0",
                kind="box",
                center=(bx - half, by, height / 2),
                half_extent=(wall, half, height),
                material="bin",
                albedo=(92, 96, 104),
                movable=False,
                is_container=True,
            ),
            ObjectSpec(
                name="bin_wall_x1",
                kind="box",
                center=(bx + half, by, height / 2),
                half_extent=(wall, half, height),
                material="bin",
                albedo=(92, 96, 104),
                movable=False,
                is_container=True,
            ),
            ObjectSpec(
                name="bin_wall_y0",
                kind="box",
                center=(bx, by - half, height / 2),
                half_extent=(half, wall, height),
                material="bin",
                albedo=(92, 96, 104),
                movable=False,
                is_container=True,
            ),
            ObjectSpec(
                name="bin_wall_y1",
                kind="box",
                center=(bx, by + half, height / 2),
                half_extent=(half, wall, height),
                material="bin",
                albedo=(92, 96, 104),
                movable=False,
                is_container=True,
            ),
        ]

        return SceneSpec(
            scene_id=f"{task.robot}-{task.seed:04d}",
            instruction=task.instruction,
            template="pick_place",
            morphology=task.morphology,
            robot=task.robot,
            camera=self.camera,
            objects=objects,
            success={
                "type": "object_inside_container",
                "object": "cube",
                "container": "bin_floor",
                "tolerance_m": 0.05,
            },
            seed=task.seed,
            notes=notes,
        )
