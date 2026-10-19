"""Command line: embodigen gen | validate | backends"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embodigen.modules.base import BACKENDS, resolve
from embodigen.modules.pack import read_episode
from embodigen.pipeline import Pipeline, PipelineConfig
from embodigen.robots import available_robots, load_robot


def _cmd_gen(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        out_dir=Path(args.out),
        threshold_mm=args.threshold,
        max_attempts=args.attempts,
        fps=args.fps,
        resolution=args.resolution,
        seed=args.seed,
        inject_drift_px=args.drift,
        pack=not args.no_pack,
    )
    pipeline = Pipeline(cfg)
    episode = pipeline.run(args.task, robot_name=args.robot)
    rep = episode.fidelity

    print(f"episode      {episode.episode_id}")
    print(f"instruction  {episode.task.instruction}")
    print(f"robot        {load_robot(args.robot)['morphology']} / {args.robot}")
    print(f"frames       {len(episode.gt)} @ {cfg.fps:g} fps, {cfg.resolution}x{cfg.resolution}")
    print(f"attempts     {episode.provenance.attempts} (seeds {episode.provenance.seeds})")
    print(f"fidelity     p95 {rep.p95_error_mm:8.2f} mm  max {rep.max_error_mm:8.2f} mm"
          f"  threshold {rep.threshold_mm:g} mm  -> {'PASS' if rep.passed else 'FAIL'}")
    if episode.scene.notes:
        print(f"scene notes  {'; '.join(episode.scene.notes)}")
    if hasattr(episode, "root"):
        print(f"written      {episode.root}")
    return 0 if rep.passed else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.episode)
    scene, gt, rendered = read_episode(root)
    validator = resolve("validate", args.backend)()
    report = validator.score(rendered, gt, scene, args.threshold)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.passed else 1


def _cmd_backends(_: argparse.Namespace) -> int:
    for stage, entries in BACKENDS.items():
        print(f"{stage:11s} {', '.join(sorted(entries))}")
    print(f"{'robots':11s} {', '.join(available_robots())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embodigen",
        description="Task description in, label-exact photorealistic dataset out.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="run the pipeline end to end")
    g.add_argument("--task", required=True, help='e.g. "把红色方块放进盒子"')
    g.add_argument("--robot", default="franka", help="franka | mobile | humanoid")
    g.add_argument("--out", default="out", help="output directory")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--fps", type=float, default=8.0)
    g.add_argument("--resolution", type=int, default=64)
    g.add_argument("--threshold", type=float, default=12.0, help="fidelity threshold in mm")
    g.add_argument("--attempts", type=int, default=3)
    g.add_argument(
        "--drift",
        type=float,
        default=3.0,
        help="pixels of simulated geometry drift on the first render attempt (0 disables)",
    )
    g.add_argument("--no-pack", action="store_true", help="skip writing the dataset")
    g.set_defaults(func=_cmd_gen)

    v = sub.add_parser("validate", help="re-check a packed episode")
    v.add_argument("--episode", required=True)
    v.add_argument("--threshold", type=float, default=12.0)
    v.add_argument("--backend", default="keypoint")
    v.set_defaults(func=_cmd_validate)

    b = sub.add_parser("backends", help="list registered backends and robots")
    b.set_defaults(func=_cmd_backends)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
