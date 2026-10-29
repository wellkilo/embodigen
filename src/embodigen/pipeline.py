"""Orchestration: instruction -> certified episode.

Two properties matter here and are easy to lose when these stages are wired
up ad hoc:

* every retry is recorded, so an episode can be audited (which seed, which
  condition weights, how many attempts);
* a failed validation never silently reaches the dataset -- it either retries
  with stronger geometry conditioning or the episode is reported as failed
  and packed under ``rejected/`` (kept for auditing, never under
  ``episodes/``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
    inject_drift_mm: float = 25.0
    drift_frames: tuple[int, int] = (6, 16)
    grain: float = 1.0
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
            "inject_drift_mm": self.inject_drift_mm,
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
            drift = cfg.inject_drift_mm if attempt == 0 else 0.0
            photorealizer = self.photorealizer_cls(
                drift_mm=drift, drift_frames=cfg.drift_frames, grain=cfg.grain
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
                validator=self.validator.name,
            ),
        )

        if cfg.pack:
            # Only certified episodes enter the dataset. Rejected ones are kept
            # next to it so a failure can still be inspected and audited.
            bucket, other = ("episodes", "rejected") if report.passed else ("rejected", "episodes")
            stale = Path(cfg.out_dir) / other / episode.episode_id
            if stale.is_dir():
                shutil.rmtree(stale)
            episode.root = self.packer.pack(episode, Path(cfg.out_dir) / bucket)  # type: ignore[attr-defined]
        return episode

    def run_batch(
        self,
        instruction: str,
        robot_name: str = "franka",
        count: int = 1,
        workers: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate ``count`` episodes on consecutive seeds.

        Workers return summaries rather than episodes: frames stay on disk,
        so a batch costs the same memory as a single episode.
        """
        seeds = [self.config.seed + i for i in range(count)]
        jobs = [(self.config, instruction, robot_name, s) for s in seeds]
        if workers <= 1 or count <= 1:
            rows = [_run_one(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, count)) as pool:
                rows = list(pool.map(_run_one, *zip(*jobs)))
        if self.config.pack:
            write_index(Path(self.config.out_dir), rows, instruction, robot_name, self.config)
        return rows


def summarize(episode: Episode) -> dict[str, Any]:
    rep = episode.fidelity
    root = getattr(episode, "root", None)
    return {
        "episode_id": episode.episode_id,
        "seed": episode.task.seed,
        "passed": rep.passed,
        "p95_error_mm": round(rep.p95_error_mm, 4),
        "max_error_mm": round(rep.max_error_mm, 4),
        "attempts": episode.provenance.attempts,
        "frames": len(episode.gt),
        "path": str(root) if root is not None else None,
    }


def _run_one(config: PipelineConfig, instruction: str, robot_name: str, seed: int) -> dict[str, Any]:
    # Top-level so it pickles into a worker process.
    return summarize(Pipeline(config).run(instruction, robot_name=robot_name, seed=seed))


def _relative(path: str | None, base: Path) -> str | None:
    """Episode paths relative to the dataset root, so a copied dataset still resolves."""
    if not path or not Path(path).is_absolute():
        return path
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return path


def write_index(
    out_dir: Path,
    rows: list[dict[str, Any]],
    instruction: str,
    robot_name: str,
    config: PipelineConfig,
) -> Path:
    """Merge a batch into ``<out>/index.json`` -- the dataset-level ledger."""
    path = out_dir / "index.json"
    index: dict[str, Any] = {"episodes": []}
    if path.exists():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    by_id = {row["episode_id"]: row for row in index.get("episodes", [])}
    for row in rows:
        by_id[row["episode_id"]] = {**row, "instruction": instruction, "robot": robot_name}
    for row in by_id.values():  # rows merged from an older index too
        row["path"] = _relative(row.get("path"), out_dir)
    episodes = sorted(by_id.values(), key=lambda r: r["episode_id"])
    certified = sum(1 for r in episodes if r["passed"])
    index = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config.fingerprint(),
        "threshold_mm": config.threshold_mm,
        "summary": {
            "total": len(episodes),
            "certified": certified,
            "rejected": len(episodes) - certified,
            "yield": round(certified / len(episodes), 4) if episodes else 0.0,
        },
        "episodes": episodes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
