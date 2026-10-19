"""Orchestration: instruction -> certified episode.

Two properties matter here and are easy to lose when these stages are wired
up ad hoc:

* every retry is recorded, so an episode can be audited (which seed, which
  condition weights, how many attempts);
* a failed validation never silently reaches the dataset -- it either retries
  with stronger geometry conditioning or the episode is reported as failed.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodigen.modules.base import resolve
from embodigen.robots import load_robot
from embodigen.types import (
    Camera,
    Episode,
    Provenance,
    TaskSpec,
)


@dataclass
class PipelineConfig:
    out_dir: Path = Path("out")
    threshold_mm: float = 12.0
    max_attempts: int = 3
    fps: float = 8.0
    resolution: int = 64
    seed: int = 0
    inject_drift_px: float = 3.0
    drift_frames: tuple[int, int] = (6, 16)
    pack: bool = True
    scene_backend: str = "template"
    trajectory_backend: str = "waypoint"
    export_backend: str = "software"
    photoreal_backend: str = "mock"
    validate_backend: str = "keypoint"
    pack_backend: str = "lerobot"

    def fingerprint(self) -> str:
        payload = {
            "threshold_mm": self.threshold_mm,
            "fps": self.fps,
            "resolution": self.resolution,
            "seed": self.seed,
            "inject_drift_px": self.inject_drift_px,
            "backends": [
                self.scene_backend,
                self.trajectory_backend,
                self.export_backend,
                self.photoreal_backend,
                self.validate_backend,
                self.pack_backend,
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None, **overrides: Any) -> None:
        self.config = config or PipelineConfig(**overrides)
        self.scene_gen = resolve("scene", self.config.scene_backend)(
            Camera().at_resolution(self.config.resolution)
        )
        self.trajectory_gen = resolve("trajectory", self.config.trajectory_backend)(
            fps=self.config.fps
        )
        self.exporter = resolve("export", self.config.export_backend)()
        self.photorealizer_cls = resolve("photoreal", self.config.photoreal_backend)
        self.validator = resolve("validate", self.config.validate_backend)()
        self.packer = resolve("pack", self.config.pack_backend)()

    def run(self, instruction: str, robot_name: str = "franka", seed: int | None = None) -> Episode:
        cfg = self.config
        seed = cfg.seed if seed is None else seed
        rng = random.Random(seed)
        robot = load_robot(robot_name)
        task = TaskSpec(
            instruction=instruction,
            morphology=robot["morphology"],
            robot=robot_name,
            seed=seed,
        )

        scene = self.scene_gen.generate(task, robot, rng)
        trajectory = self.trajectory_gen.plan(scene, robot)
        gt = self.exporter.export(scene, trajectory)

        seeds: list[int] = []
        attempts = 0
        report = None
        rendered = None
        weights: dict[str, float] = {}

        for attempt in range(cfg.max_attempts):
            attempts = attempt + 1
            seeds.append(seed + attempt)
            # Each retry leans harder on geometry, per the design doc.
            weights = {
                "geometry": 1.0 + 0.6 * attempt,
                "depth": 1.0 + 0.3 * attempt,
                "segmentation": 1.0 + 0.3 * attempt,
                "camera": 1.0,
            }
            drift = cfg.inject_drift_px if attempt == 0 else 0.0
            photorealizer = self.photorealizer_cls(
                drift_px=drift, drift_frames=cfg.drift_frames
            )
            rendered = photorealizer.render(gt, scene, weights, seed + attempt)
            report = self.validator.score(rendered, gt, scene, cfg.threshold_mm)
            if report.passed:
                break

        assert rendered is not None and report is not None
        episode = Episode(
            episode_id=f"{robot_name}-{seed:04d}",
            task=task,
            scene=scene,
            trajectory=trajectory,
            gt=gt,
            photoreal=rendered,
            fidelity=report,
            provenance=Provenance(
                engine=self.exporter.name,
                photorealizer=self.photorealizer_cls.name,
                config_hash=cfg.fingerprint(),
                seeds=seeds,
                threshold_mm=cfg.threshold_mm,
                condition_weights=weights,
                attempts=attempts,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

        if cfg.pack:
            episode.root = self.packer.pack(episode, Path(cfg.out_dir) / "episodes")  # type: ignore[attr-defined]
        return episode
