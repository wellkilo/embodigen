"""Command line: embodigen gen | validate | preview | index | backends"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embodigen.modules.base import BACKENDS, resolve
from embodigen.modules.pack import read_episode
from embodigen.pipeline import Pipeline, PipelineConfig, default_workers, summarize, write_index
from embodigen.robots import available_robots, load_robot


def _cmd_gen(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        out_dir=Path(args.out),
        threshold_mm=args.threshold,
        max_attempts=args.attempts,
        fps=args.fps,
        resolution=args.resolution,
        seed=args.seed,
        inject_drift_mm=args.drift,
        pack=not args.no_pack,
    )
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    pipeline = Pipeline(cfg)
    if args.episodes > 1:
        return _gen_batch(pipeline, args)

    episode = pipeline.run(args.task, robot_name=args.robot)
    if cfg.pack:
        write_index(cfg.out_dir, [summarize(episode)], args.task, args.robot, cfg)
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
        label = "written" if rep.passed else "rejected"
        print(f"{label:12s} {episode.root}")
    return 0 if rep.passed else 1


def _gen_batch(pipeline: Pipeline, args: argparse.Namespace) -> int:
    cfg = pipeline.config
    workers = args.workers or default_workers()
    print(f"generating   {args.episodes} episodes · robot {args.robot} · "
          f"{cfg.resolution}px · {workers} worker(s)")
    rows = pipeline.run_batch(args.task, args.robot, count=args.episodes, workers=workers)
    for row in rows:
        verdict = "PASS" if row["passed"] else "REJECT"
        print(f"  {row['episode_id']:16s} p95 {row['p95_error_mm']:7.2f} mm  "
              f"attempts {row['attempts']}  {verdict}")
    certified = sum(1 for r in rows if r["passed"])
    print(f"certified    {certified}/{len(rows)} ({certified / len(rows):.0%} yield)")
    if cfg.pack:
        print(f"index        {cfg.out_dir / 'index.json'}")
    # Rejection is the validator doing its job; only an empty batch is an error.
    return 0 if certified else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.episode)
    scene, gt, rendered = read_episode(root)
    validator = resolve("validate", args.backend)()
    report = validator.score(rendered, gt, scene, args.threshold)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.passed else 1


def _cmd_preview(args: argparse.Namespace) -> int:
    from embodigen.preview import render_preview

    root = Path(args.episode)
    out = Path(args.out) if args.out else root / "preview.gif"
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    path, frames, w, h = render_preview(
        root, out, channels=channels, fps=args.fps, scale=args.scale, max_frames=args.max_frames
    )
    print(f"wrote {path} — {frames} frames, {w}x{h}, {path.stat().st_size / 1024:.0f} KB")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    path = Path(args.out) / "index.json"
    if not path.exists():
        raise SystemExit(f"no index at {path}; run `embodigen gen --out {args.out}` first")
    index = json.loads(path.read_text(encoding="utf-8"))
    summary = index["summary"]
    print(f"{'episode':16s} {'robot':9s} {'p95 mm':>8s} {'attempts':>8s}  verdict")
    for row in index["episodes"]:
        verdict = "PASS" if row["passed"] else "REJECT"
        print(f"{row['episode_id']:16s} {row.get('robot', '?'):9s} "
              f"{row['p95_error_mm']:8.2f} {row['attempts']:8d}  {verdict}")
    print(f"\n{summary['certified']}/{summary['total']} certified "
          f"({summary['yield']:.0%} yield), threshold {index['threshold_mm']:g} mm")
    return 0


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
        default=25.0,
        help="millimetres of simulated geometry drift on the first render attempt (0 disables)",
    )
    g.add_argument("--no-pack", action="store_true", help="skip writing the dataset")
    g.add_argument("--episodes", "-n", type=int, default=1,
                   help="number of episodes, on consecutive seeds starting at --seed")
    g.add_argument("--workers", "-j", type=int, default=0,
                   help="parallel processes for --episodes > 1 (default: CPU count - 1)")
    g.set_defaults(func=_cmd_gen)

    v = sub.add_parser("validate", help="re-check a packed episode")
    v.add_argument("--episode", required=True)
    v.add_argument("--threshold", type=float, default=12.0)
    v.add_argument("--backend", default="keypoint")
    v.set_defaults(func=_cmd_validate)

    pv = sub.add_parser("preview", help="render a packed episode to an animated GIF")
    pv.add_argument("--episode", required=True)
    pv.add_argument("--out", default=None, help="default: <episode>/preview.gif")
    pv.add_argument("--channels", default="white,photoreal,depth,seg",
                    help="comma-separated subset of white,photoreal,depth,seg")
    pv.add_argument("--fps", type=float, default=8.0)
    pv.add_argument("--scale", type=int, default=0, help="integer upscale (default: ~160 px panels)")
    pv.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    pv.set_defaults(func=_cmd_preview)

    ix = sub.add_parser("index", help="summarise every episode under an output directory")
    ix.add_argument("--out", default="out")
    ix.set_defaults(func=_cmd_index)

    b = sub.add_parser("backends", help="list registered backends and robots")
    b.set_defaults(func=_cmd_backends)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
