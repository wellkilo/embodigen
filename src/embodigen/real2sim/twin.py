"""A digital twin of a tabletop pick-and-place scene, and variants of it for training data.

The twin file records what was measured in the real scene, in the robot's table frame (metres,
+Z up): where the cube and the bin are, the cube's size and the colours that matter. It does not
pretend to be a reconstruction: it says how it was obtained (``source``) and whether the numbers
are measured by a person or estimated by a model, and downstream data carries that label.

Sim2real data comes from randomising around the twin (``jitter``): object positions, colours and
the camera move by small amounts, every variant is checked for reachability, and each one still
passes through the pipeline's own plan -> render -> fidelity gate, so only variants whose rendered
pixels agree with their geometry are certified.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodigen.geometry import CameraModel, dist
from embodigen.modules.scene import (
    BIN_ALBEDO,
    BIN_HALF,
    TABLE_ALBEDO,
    pick_place_objects,
    pick_place_success,
)
from embodigen.types import Camera, SceneSpec, TaskSpec, Vec3

FORMAT = "embodigen-twin-v1"
CUBE_HALF_RANGE = (0.015, 0.03)  # the grasp planner's offsets assume a cube of this order
REACH_FRACTION = 0.7  # of the robot's reach, for both the cube and the bin
VIEW_MARGIN_PX = 4  # objects must project this far inside the image


def _rgb(value: Any) -> tuple[int, int, int]:
    rgb = tuple(int(v) for v in value)
    if len(rgb) != 3 or any(not 0 <= v <= 255 for v in rgb):
        raise ValueError(f"colour must be three values in 0..255, got {value!r}")
    return rgb  # type: ignore[return-value]


@dataclass
class Twin:
    cube_xy: tuple[float, float]
    bin_xy: tuple[float, float]
    cube_half: float = 0.02
    cube_rgb: tuple[int, int, int] = (198, 62, 58)
    bin_rgb: tuple[int, int, int] = BIN_ALBEDO
    table_rgb: tuple[int, int, int] = TABLE_ALBEDO
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "manual", "estimated": False})

    @property
    def twin_id(self) -> str:
        payload = json.dumps({k: v for k, v in self.to_json().items() if k != "source"}, sort_keys=True)
        return "twin-" + hashlib.sha256(payload.encode()).hexdigest()[:10]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("cube_xy", "bin_xy", "cube_rgb", "bin_rgb", "table_rgb"):
            d[key] = list(d[key])
        return {"format": FORMAT, **d}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Twin":
        if data.get("format") != FORMAT:
            raise SystemExit(f"not a twin file (format {data.get('format')!r}, expected {FORMAT})")
        try:
            return cls(
                cube_xy=(float(data["cube_xy"][0]), float(data["cube_xy"][1])),
                bin_xy=(float(data["bin_xy"][0]), float(data["bin_xy"][1])),
                cube_half=float(data.get("cube_half", 0.02)),
                cube_rgb=_rgb(data.get("cube_rgb", (198, 62, 58))),
                bin_rgb=_rgb(data.get("bin_rgb", BIN_ALBEDO)),
                table_rgb=_rgb(data.get("table_rgb", TABLE_ALBEDO)),
                source=dict(data.get("source") or {}),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SystemExit(f"bad twin file: {exc!r}") from None


def load_twin(path: str | Path) -> Twin:
    try:
        return Twin.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read twin {path}: {exc}") from None


def write_twin(twin: Twin, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(twin.to_json(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def problems(twin: Twin, robot: dict, camera: Camera | None = None) -> list[str]:
    """Reasons the planner could not run this twin (empty: it can)."""
    out = []
    base: Vec3 = tuple(robot.get("base", (0.0, 0.0, 0.0)))  # type: ignore[assignment]
    reach = float(robot.get("reach", 0.85)) * REACH_FRACTION
    cube: Vec3 = (twin.cube_xy[0], twin.cube_xy[1], twin.cube_half)
    bin_: Vec3 = (twin.bin_xy[0], twin.bin_xy[1], 0.0)
    if not CUBE_HALF_RANGE[0] <= twin.cube_half <= CUBE_HALF_RANGE[1]:
        out.append(f"cube half-size {twin.cube_half:g} m outside {CUBE_HALF_RANGE[0]:g}..{CUBE_HALF_RANGE[1]:g}")
    if dist(cube, base) > reach:
        out.append(f"cube is {dist(cube, base):.2f} m from the robot base, beyond {reach:.2f} m")
    if dist(bin_, base) > reach:
        out.append(f"bin is {dist(bin_, base):.2f} m from the robot base, beyond {reach:.2f} m")
    gap = max(abs(cube[0] - bin_[0]), abs(cube[1] - bin_[1]))
    if gap < BIN_HALF + twin.cube_half + 0.01:
        out.append(f"cube overlaps the bin (centres {gap * 100:.0f} cm apart)")
    cam = CameraModel(camera or Camera())
    cw, ch = (camera or Camera()).width, (camera or Camera()).height
    for name, p in (("cube", cube), ("bin", bin_)):
        u, v, z = cam.project(p)
        if z <= 0 or not (VIEW_MARGIN_PX <= u <= cw - VIEW_MARGIN_PX and VIEW_MARGIN_PX <= v <= ch - VIEW_MARGIN_PX):
            out.append(f"{name} is outside the camera view")
    return out


def _nudge(rng: random.Random, rgb: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in rgb)  # type: ignore[return-value]


class TwinSceneGenerator:
    """Scenes from a twin: exactly the twin at ``jitter`` 0, randomised variants above it.

    ``jitter`` is the largest shift in metres of an object (positions), the camera eye moves by a
    fifth of it, and colours by ``jitter * 600`` levels (0.03 m -> +-18 of 255). Variants that the
    robot could not reach, or that leave the view, are resampled; if none is found the exact
    twin is used and the scene says so."""

    name = "twin-scene-v0"

    def __init__(self, camera: Camera | None = None, twin: Twin | None = None, jitter: float = 0.0,
                 max_resamples: int = 32) -> None:
        if twin is None:
            raise SystemExit("the twin scene backend needs a twin: embodigen gen --twin twin.json")
        self.camera, self.twin, self.jitter, self.max_resamples = camera or Camera(), twin, jitter, max_resamples

    def generate(self, task: TaskSpec, robot: dict, rng: random.Random) -> SceneSpec:
        twin, camera, notes = self.twin, self.camera, []
        if self.jitter > 0:
            for attempt in range(self.max_resamples):
                j, shade = self.jitter, int(round(self.jitter * 600))
                cand = Twin(
                    cube_xy=(twin.cube_xy[0] + rng.uniform(-j, j), twin.cube_xy[1] + rng.uniform(-j, j)),
                    bin_xy=(twin.bin_xy[0] + rng.uniform(-j, j), twin.bin_xy[1] + rng.uniform(-j, j)),
                    cube_half=twin.cube_half,
                    cube_rgb=_nudge(rng, twin.cube_rgb, shade),
                    bin_rgb=_nudge(rng, twin.bin_rgb, shade),
                    table_rgb=_nudge(rng, twin.table_rgb, shade),
                    source=twin.source,
                )
                eye = tuple(e + rng.uniform(-j, j) / 5 for e in self.camera.eye)
                cam = Camera(**{**self.camera.to_dict(), "eye": eye})
                if not problems(cand, robot, cam):
                    twin, camera = cand, cam
                    notes.append(f"randomised around the twin (jitter {self.jitter:g} m)"
                                 + (f", resampled {attempt}x" if attempt else ""))
                    break
            else:
                notes.append("no reachable variant found; used the exact twin")
        found = problems(twin, robot, camera)
        if found:
            raise SystemExit("this twin cannot be run by the planner: " + "; ".join(found))
        notes += [f"twin={self.twin.twin_id}",
                  "twin-source=" + ("model-estimated from a photo" if self.twin.source.get("estimated") else "measured by hand")]
        cube: Vec3 = (twin.cube_xy[0], twin.cube_xy[1], twin.cube_half)
        bin_: Vec3 = (twin.bin_xy[0], twin.bin_xy[1], 0.0)
        return SceneSpec(
            scene_id=f"{task.robot}-{task.seed:04d}",
            instruction=task.instruction,
            template="pick_place",
            morphology=task.morphology,
            robot=task.robot,
            camera=camera,
            objects=pick_place_objects(cube, bin_, twin.cube_rgb, twin.cube_half, twin.bin_rgb, twin.table_rgb),
            success=pick_place_success(),
            seed=task.seed,
            notes=notes,
        )
