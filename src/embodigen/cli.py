"""Command line.

Robot track:          gen | validate | preview | index
Virtual-world track:  worlds | rollout | ingest | curate | build | ledger
Both:                 backends
"""

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

    if args.recording:
        from embodigen.gameplay.preview import render_recording

        root = Path(args.recording)
        out = Path(args.out) if args.out else root / "preview.gif"
        path, frames, w, h = render_recording(root, out, max_turns=args.max_frames, scale=args.scale)
        print(f"wrote {path} — {frames} turns, {w}x{h}, {path.stat().st_size / 1024:.0f} KB")
        return 0
    if not args.episode:
        raise SystemExit("preview needs --episode (robot track) or --recording (virtual-world track)")
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


# -- virtual-world track ------------------------------------------------------

def _registry(args: argparse.Namespace):
    from embodigen.gameplay.registry import WorldRegistry

    return WorldRegistry.load(Path(args.worlds_dir) if getattr(args, "worlds_dir", None) else None)


def _cmd_worlds(args: argparse.Namespace) -> int:
    registry = _registry(args)
    width = max([18, *(len(label) for label in registry.worlds)])
    print(f"{'world':{width}s} {'cat':4s} {'view':5s} {'engine':9s} aliases")
    for spec in sorted(registry.worlds.values(), key=lambda s: (not s.engine, s.label)):
        print(f"{spec.label:{width}s} {spec.category:4s} {spec.view:5s} {spec.engine or '-':9s} "
              f"{', '.join(spec.aliases) or '-'}")
    return 0


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h or w)


def _cmd_rollout(args: argparse.Namespace) -> int:
    from embodigen.gameplay.pipeline import default_workers, run_rollouts

    registry = _registry(args)
    worlds = [w.strip() for w in args.world.split(",") if w.strip()]
    if worlds == ["all"]:
        worlds = [w for w, spec in registry.worlds.items() if spec.engine == "raycast"]
    workers = args.workers or default_workers()
    rows = run_rollouts(
        registry, worlds, args.episodes, args.seconds, Path(args.out), seed=args.seed,
        profile=args.profile, lag_ms=args.lag_ms, drift_ms_per_min=args.drift_ms_per_min, size=_size(args.resolution),
        streams=args.streams, gt=not args.no_gt, workers=workers,
    )
    for row in rows:
        print(f"  {row['recording_id']:34s} {row['duration']:6.1f}s  stream {row['stream']}")
    print(f"recorded     {len(rows)} recordings, {sum(r['duration'] for r in rows) / 60:.1f} min -> {args.out}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from embodigen.gameplay.pipeline import default_workers, run_ingest

    workers = args.workers or default_workers()
    config_dir = Path(args.worlds_dir) if getattr(args, "worlds_dir", None) else None
    rows = run_ingest(args.source, Path(args.input), Path(args.out), _registry(args), workers, config_dir)
    for row in rows:
        hint = f"  split={row['split_hint']}" if row["split_hint"] else ""
        print(f"  {row['recording_id']:40s} {row['duration']:7.1f}s  {row['stream']}{hint}")
    print(f"ingested     {len(rows)} {args.source} recordings -> {args.out}")
    return 0 if rows else 1


def _cmd_curate(args: argparse.Namespace) -> int:
    from embodigen.gameplay.pipeline import curate, default_workers
    from embodigen.gameplay.quality import QualityConfig

    cfg = QualityConfig(realign=not args.no_realign, alignment=not args.no_alignment,
                        video_backend="" if args.no_video else args.video_backend)
    ledger = curate(Path(args.raw), Path(args.out), cfg, workers=args.workers or default_workers())
    for line in _curate_lines(Path(args.out)):
        print(line)
    s = ledger["summary"]
    print(f"certified    {s['certified']}/{s['recordings']} ({s['yield']:.0%} yield), "
          f"{s['hours_certified'] * 60:.1f} min, {s['realigned']} realigned -> {args.out}")
    return 0 if s["certified"] else 1


def _curate_lines(curated: Path) -> list[str]:
    lines = []
    for line in (curated / "recordings.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        al = row.get("alignment", {})
        lag = al.get("before", al).get("lag_ms")
        align = al.get("status", "-") + (f" ({lag:+.0f} ms)" if lag and al.get("status") in ("misaligned", "drift") else "")
        if row["realigned_ms"]:
            align = f"realigned {row['realigned_ms']:+.0f} ms"
        verdict = "PASS" if row["passed"] else "REJECT  " + "; ".join(row["reasons"])
        lines.append(f"  {row['recording_id']:34s} {align:24s} {verdict}")
    return lines


def _cmd_build(args: argparse.Namespace) -> int:
    from embodigen.gameplay.pipeline import build, load_study

    study = load_study(Path(args.study))
    out = Path(args.out) / study.name if not args.flat else Path(args.out)
    build(Path(args.curated), study, _registry(args), out, annotator_name=args.annotator)
    _print_ledger(out / "ledger.json")
    return 0


def _print_ledger(path: Path) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("stage") == "curate":
        width = max([18, *(len(world) for world in ledger["worlds"])])
        print(f"{'world':{width}s} {'source':>8s} {'recs':>5s} {'certified':>9s} {'pass':>6s} {'minutes':>8s} "
              f"{'clip':>6s}  rejections")
        for world, w in ledger["worlds"].items():
            reasons = ", ".join(f"{k}x{v}" for k, v in w["reasons"].items()) or "-"
            source = ",".join(s.split(":")[0] for s in w.get("sources", [])) or "-"
            print(f"{world:{width}s} {source:>8s} {w['recordings']:5d} {w['certified']:9d} {w['pass_rate']:6.0%} "
                  f"{w['hours_certified'] * 60:8.1f} {w.get('mouse_clipped_frac', 0.0):6.1%}  {reasons}")
        print("clip = share of 200 ms packets whose mouse motion hit +-1000 (tune sources.<name>.mouse_scale)")
        counts = ledger.get("summary", {}).get("alignment")
        if counts:
            print("alignment: " + ", ".join(f"{k} {v}" for k, v in counts.items())
                  + "  (inconclusive/skipped = not verifiable from pixels, not rejected)")
        return
    pool = ledger["pool"]
    print(f"study {ledger['study']}  config {ledger['config_hash']}  annotator {ledger['annotator']}")
    print(f"pool  {pool['windows']} windows, {pool['footage_hours'] * 60:.1f} min of footage "
          f"(budgets count {pool['budget_unit']}), D={pool['diversity']['hill_d']:.2f}"
          f" over {pool['diversity']['worlds']} worlds")
    for level, v in ledger["validation"].items():
        print(f"val   {level:9s} {v['clips']:4d} clips  {len(v['worlds'])} worlds  {v['recordings']} recordings")
    print(f"\n{'mix':4s} {'budget h':>8s} {'footage':>7s} {'samples':>7s} {'worlds':>6s} {'Hill D':>6s} "
          f"{'1/Simp':>6s} {'max':>6s} {'anchor':>6s} {'think':>6s} {'short h':>7s}")
    for g in ledger["grid"]:
        print(f"{g['mixture']:4s} {g['budget_hours']:8g} {g['footage_hours']:7.3f} {g['samples']:7d} {g['worlds']:6d} "
              f"{g['hill_d']:6.2f} {g['inverse_simpson']:6.2f} {g['largest_share']:6.1%} "
              f"{g['anchor_share']:6.1%} {g['rationale_density']:6.1%} {g['shortfall_hours']:7.3f}")
    checks = ledger["checks"]
    print(f"\nchecks: budgets nested {checks['budgets_nested']}, train/val disjoint {checks['train_val_disjoint']}, "
          f"rationale density spread {checks['rationale_density_spread']:.2%}")
    for note in ledger.get("notes", []):
        print(f"note: {note}")


def _cmd_ledger(args: argparse.Namespace) -> int:
    path = Path(args.path)
    path = path / "ledger.json" if path.is_dir() else path
    if not path.exists():
        raise SystemExit(f"no ledger at {path}")
    _print_ledger(path)
    return 0


def _cmd_backends(_: argparse.Namespace) -> int:
    for stage, entries in BACKENDS.items():
        print(f"{stage:11s} {', '.join(sorted(entries))}")
    print(f"{'robots':11s} {', '.join(available_robots())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embodigen",
        description="Label-exact data for embodied agents: robot episodes and virtual-world gameplay.",
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

    pv = sub.add_parser("preview", help="render a packed episode or a raw recording to an animated GIF")
    pv.add_argument("--episode", default=None, help="a packed robot episode")
    pv.add_argument("--recording", default=None, help="a raw gameplay recording directory")
    pv.add_argument("--out", default=None, help="default: <episode or recording>/preview.gif")
    pv.add_argument("--channels", default="white,photoreal,depth,seg",
                    help="comma-separated subset of white,photoreal,depth,seg")
    pv.add_argument("--fps", type=float, default=8.0)
    pv.add_argument("--scale", type=int, default=0, help="integer upscale (default: ~160 px panels)")
    pv.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    pv.set_defaults(func=_cmd_preview)

    ix = sub.add_parser("index", help="summarise every episode under an output directory")
    ix.add_argument("--out", default="out")
    ix.set_defaults(func=_cmd_index)

    wd = argparse.ArgumentParser(add_help=False)
    wd.add_argument("--worlds-dir", default=None, help="world registry (default: configs/worlds)")

    wl = sub.add_parser("worlds", parents=[wd], help="list registered virtual worlds")
    wl.set_defaults(func=_cmd_worlds)

    ro = sub.add_parser("rollout", parents=[wd], help="play worlds with the scripted player and record them")
    ro.add_argument("--world", required=True, help="comma-separated world labels, or 'all' for the raycast worlds")
    ro.add_argument("--episodes", "-n", type=int, default=2, help="recordings per world")
    ro.add_argument("--seconds", type=float, default=30.0)
    ro.add_argument("--out", default="data/raw")
    ro.add_argument("--seed", type=int, default=0)
    ro.add_argument("--profile", default="player", help="player | afk | distracted")
    ro.add_argument("--lag-ms", type=float, default=0.0,
                    help="shift the input log against the video (desync demo; the gate should catch it)")
    ro.add_argument("--drift-ms-per-min", type=float, default=0.0,
                    help="let the input log drift against the video (clock-drift demo; not correctable)")
    ro.add_argument("--resolution", default="160x100", help="WxH (ViZDoom needs one of its modes, e.g. 160x120)")
    ro.add_argument("--streams", type=int, default=2, help="distinct player streams per world")
    ro.add_argument("--no-gt", action="store_true", help="skip depth/label export")
    ro.add_argument("--workers", "-j", type=int, default=0)
    ro.set_defaults(func=_cmd_rollout)

    ig = sub.add_parser("ingest", parents=[wd], help="convert a dataset's recordings into the raw layout")
    ig.add_argument("--source", required=True, help="vpt | d2e | p2p")
    ig.add_argument("--input", required=True)
    ig.add_argument("--out", default="data/raw")
    ig.add_argument("--workers", "-j", type=int, default=0)
    ig.set_defaults(func=_cmd_ingest)

    cu = sub.add_parser("curate", help="align, canonicalise and gate raw recordings")
    cu.add_argument("--raw", default="data/raw")
    cu.add_argument("--out", default="data/curated")
    cu.add_argument("--no-realign", action="store_true", help="reject desynced recordings instead of correcting them")
    cu.add_argument("--no-alignment", action="store_true", help="skip the action-frame alignment check")
    cu.add_argument("--video-backend", default="pyav", help="decoder for frames kept in videos (pyav)")
    cu.add_argument("--no-video", action="store_true", help="skip alignment on video-only recordings")
    cu.add_argument("--workers", "-j", type=int, default=0)
    cu.set_defaults(func=_cmd_curate)

    bu = sub.add_parser("build", parents=[wd], help="split, mix, annotate and shard a curated pool")
    bu.add_argument("--curated", default="data/curated")
    bu.add_argument("--study", default="configs/studies/reference.toml")
    bu.add_argument("--out", default="data/mixes", help="written to <out>/<study name>/")
    bu.add_argument("--flat", action="store_true", help="write directly into --out")
    bu.add_argument("--annotator", default=None, help="override the study's annotator backend")
    bu.set_defaults(func=_cmd_build)

    lg = sub.add_parser("ledger", help="print a curate or build ledger")
    lg.add_argument("path", help="a curated/ or mixes/<study>/ directory, or a ledger.json")
    lg.set_defaults(func=_cmd_ledger)

    b = sub.add_parser("backends", help="list registered backends and robots")
    b.set_defaults(func=_cmd_backends)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
