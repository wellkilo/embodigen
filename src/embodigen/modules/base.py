"""Stage protocols.

A module satisfies a protocol by shape, not inheritance, so a third-party
backend does not have to import our classes to plug in.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

from embodigen.types import (
    Episode,
    FidelityReport,
    FrameGT,
    PhotorealResult,
    SceneSpec,
    TaskSpec,
    Trajectory,
)


class SceneGenerator(Protocol):
    name: str

    def generate(self, task: TaskSpec, robot: dict, rng: random.Random) -> SceneSpec: ...


class TrajectoryGen(Protocol):
    name: str

    def plan(self, scene: SceneSpec, robot: dict) -> Trajectory: ...


class GTExporter(Protocol):
    name: str

    def export(self, scene: SceneSpec, trajectory: Trajectory) -> list[FrameGT]: ...


class Photorealizer(Protocol):
    name: str

    def render(
        self,
        gt: list[FrameGT],
        scene: SceneSpec,
        condition_weights: dict[str, float],
        seed: int,
    ) -> PhotorealResult: ...


class FidelityValidator(Protocol):
    name: str

    def score(
        self,
        rendered: PhotorealResult,
        gt: list[FrameGT],
        scene: SceneSpec,
        threshold_mm: float,
    ) -> FidelityReport: ...


class DatasetPacker(Protocol):
    name: str

    def pack(self, episode: Episode, out_dir: Path) -> Path: ...


# Map of backend name -> import path, filled in as real backends land.
BACKENDS: dict[str, dict[str, str]] = {
    "scene": {
        "template": "embodigen.modules.scene:TemplateSceneGenerator",
    },
    "trajectory": {
        "waypoint": "embodigen.modules.trajectory:WaypointTrajectoryGen",
    },
    "export": {
        "software": "embodigen.modules.export:SoftwareGTExporter",
    },
    "photoreal": {
        "mock": "embodigen.modules.photoreal:MockPhotorealizer",
    },
    "validate": {
        "keypoint": "embodigen.modules.validate:KeypointFidelityValidator",
    },
    "pack": {
        "lerobot": "embodigen.modules.pack:LeRobotPacker",
    },
    # Virtual-world track (embodigen.gameplay).
    "env": {
        "raycast": "embodigen.gameplay.envs.raycast:RaycastWorld",
        "vizdoom": "embodigen.gameplay.envs.vizdoom_env:VizDoomEnv",
    },
    "source": {
        "vpt": "embodigen.gameplay.sources.vpt:VPTSource",
        "d2e": "embodigen.gameplay.sources.d2e:D2ESource",
        "p2p": "embodigen.gameplay.sources.p2p:P2PSource",
    },
    "annotate": {
        "mock": "embodigen.gameplay.annotate:MockAnnotator",
        "llm": "embodigen.gameplay.llm:LLMAnnotator",
    },
    "video": {
        "pyav": "embodigen.gameplay.video:PyAVFrames",
    },
}


def resolve(stage: str, backend: str):
    """Import a backend by name, with an error that says what to install."""
    try:
        target = BACKENDS[stage][backend]
    except KeyError:
        known = ", ".join(sorted(BACKENDS.get(stage, {}))) or "<none>"
        raise SystemExit(
            f"unknown {stage} backend {backend!r}; available: {known}. "
            f"Real backends are registered in embodigen/modules/base.py."
        ) from None
    module_path, _, attr = target.partition(":")
    try:
        module = __import__(module_path, fromlist=[attr])
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise SystemExit(
            f"backend {backend!r} needs dependencies that are not installed: {exc}. "
            f"Try: pip install 'embodigen[{stage}]'"
        ) from exc
    return getattr(module, attr)
