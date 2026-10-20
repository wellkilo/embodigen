"""Generate every asset the docs page needs, straight from the pipeline.

Nothing on the showcase page is hand-drawn: the sprite sheets, the
before/after pair and the fidelity numbers all come from real runs, so the
page cannot drift away from what the code actually does.

    python scripts/build_docs_assets.py --out docs/assets
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodigen.modules.scene import TemplateSceneGenerator  # noqa: E402
from embodigen.modules.trajectory import WaypointTrajectoryGen  # noqa: E402
from embodigen.pipeline import Pipeline, PipelineConfig  # noqa: E402
from embodigen.pngio import write_png  # noqa: E402
from embodigen.robots import load_robot  # noqa: E402
from embodigen.types import Camera, TaskSpec  # noqa: E402

SHEET_COLS = 7
ROOT = Path(__file__).resolve().parents[1]


def sprite_sheet(frames: list[bytes], size: int, cols: int = SHEET_COLS) -> tuple[bytes, int, int]:
    rows = (len(frames) + cols - 1) // cols
    width, height = cols * size, rows * size
    canvas = bytearray(width * height * 3)
    for index, rgb in enumerate(frames):
        cx, cy = index % cols, index // cols
        for v in range(size):
            src = (v * size) * 3
            dst = ((cy * size + v) * width + cx * size) * 3
            canvas[dst : dst + size * 3] = rgb[src : src + size * 3]
    return bytes(canvas), width, height


def run(resolution: int, fps: float, drift: float, seed: int = 0, robot: str = "franka"):
    pipeline = Pipeline(
        PipelineConfig(
            out_dir=Path("out/docs"),
            resolution=resolution,
            fps=fps,
            max_attempts=1,
            inject_drift_mm=drift,
            seed=seed,
            pack=False,
        )
    )
    return pipeline.run("把红色方块放进盒子", robot_name=robot, seed=seed)


def metrics_of(episode, resolution: int, fps: float, drift: float) -> dict:
    rep = episode.fidelity
    return {
        "resolution": resolution,
        "fps": fps,
        "drift_mm": drift,
        "frames": len(episode.gt),
        "attempts": episode.provenance.attempts,
        "passed": rep.passed,
        "threshold_mm": rep.threshold_mm,
        "mean_mm": round(rep.mean_error_mm, 2),
        "p95_mm": round(rep.p95_error_mm, 2),
        "max_mm": round(rep.max_error_mm, 2),
        "per_frame_mm": [None if v != v else round(v, 2) for v in rep.per_frame_mm],
        "notes": rep.notes,
    }


def morphology_frames(out: Path) -> list[dict]:
    out_rows = []
    scenegen = TemplateSceneGenerator(Camera())
    planner = WaypointTrajectoryGen(fps=6)

    for index, name in enumerate(("franka", "mobile", "humanoid")):
        robot = load_robot(name)
        scene = scenegen.generate(
            TaskSpec("把红色方块放进盒子", robot["morphology"], name),
            robot,
            random.Random(0),
        )
        traj = planner.plan(scene, robot)
        # A different seed per body so the cards show genuinely different
        # layouts rather than the same scene three times.
        episode = run(160, fps=6, drift=0.0, robot=name, seed=index)
        write_png(out / f"{name}.png", 160, 160, episode.photoreal.frames[len(episode.gt) // 2].rgb)
        out_rows.append(
            {
                "name": name,
                "morphology": robot["morphology"],
                "action_dim": robot["action_dim"],
                "dof": robot["dof"],
                "reach_m": robot["reach"],
                "frames": len(traj.frames),
                "image": f"assets/{name}.png",
            }
        )
    return out_rows


def r(v, n=4):
    return [round(x, n) for x in v]


def episode_data(episode) -> dict:
    """Everything the live 3D viewer needs, taken from the engine's own state."""
    rep = episode.fidelity
    cam = episode.scene.camera
    return {
        "id": episode.episode_id,
        "instruction": episode.task.instruction,
        "fps": episode.trajectory.fps,
        "camera": {"eye": r(cam.eye), "target": r(cam.target), "fx": cam.fx, "size": cam.width},
        # The table is a 5 m slab in the engine; the viewer draws it as a floor.
        "objects": [
            {k: (r(v) if isinstance(v, tuple) and k != "albedo" else v)
             for k, v in asdict(o).items()}
            for o in episode.scene.objects
            if o.name != "table"
        ],
        "frames": [
            {
                "t": f.t,
                "phase": f.phase,
                "eef": r(f.eef),
                "gripper": round(f.gripper, 4),
                "cube": r(f.objects["cube"]),
                "action": r(f.action, 3),
                "err_mm": None if e != e else round(e, 2),
            }
            for f, e in zip(episode.gt, rep.per_frame_mm)
        ],
        "threshold_mm": rep.threshold_mm,
    }


def batch_data(out: Path, count: int = 12, size: int = 96) -> dict:
    """A real batch at default settings: every first render drifts and is caught."""
    tiles, rows = [], []
    for seed in range(count):
        pipeline = Pipeline(PipelineConfig(resolution=size, fps=6, seed=seed, pack=False))
        ep = pipeline.run("把红色方块放进盒子", seed=seed)
        tiles.append(ep.photoreal.frames[(len(ep.gt) * 2) // 3].rgb)
        rows.append({
            "id": ep.episode_id,
            "p95_mm": round(ep.fidelity.p95_error_mm, 2),
            "attempts": ep.provenance.attempts,
            "passed": ep.fidelity.passed,
        })
    cols = 6
    sheet, w, h = sprite_sheet(tiles, size, cols)
    write_png(out / "batch-sheet.png", w, h, sheet)
    certified = sum(1 for row in rows if row["passed"])
    return {
        "src": "assets/batch-sheet.png", "cols": cols, "size": size,
        "episodes": rows, "certified": certified, "total": len(rows),
    }


def cli_transcripts() -> list[dict]:
    """Run the real CLI and keep its stdout, so the page's terminal never lies."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    commands = [
        ("单条", ["gen", "--task", "把红色方块放进盒子", "--out", "out", "--resolution", "96"]),
        ("批量", ["gen", "--task", "把红色方块放进盒子", "--out", "out", "--resolution", "64",
                 "--episodes", "6", "--seed", "1", "--workers", "4"]),
        ("账本", ["index", "--out", "out"]),
        ("预览", ["preview", "--episode", "out/episodes/franka-0000"]),
    ]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, args in commands:
            proc = subprocess.run(
                [sys.executable, "-m", "embodigen.cli", *args],
                cwd=tmp, env=env, capture_output=True, text=True, check=False,
            )
            shown = " ".join(f'"{a}"' if " " in a or not a.isascii() else a for a in args)
            out.append({"label": label, "cmd": f"embodigen {shown}", "output": proc.stdout.rstrip()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/assets")
    parser.add_argument("--size", type=int, default=224, help="sprite frame size in px")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] clean run @ {args.size}px")
    clean = run(args.size, fps=6, drift=0.0)
    white, w, h = sprite_sheet([f.rgb for f in clean.gt], args.size)
    write_png(out / "white-sheet.png", w, h, white)
    photo, pw, ph = sprite_sheet([f.rgb for f in clean.photoreal.frames], args.size)
    write_png(out / "photoreal-sheet.png", pw, ph, photo)

    print(f"[2/6] drifted run @ {args.size}px (rejection path)")
    drifted = run(args.size, fps=6, drift=25.0)
    # No drift sheet: the white model is the engine's ground truth and never
    # moves. Drift happens downstream, in the photorealizer.

    print("[3/6] hero pair @ 320px")
    hero = run(320, fps=4, drift=0.0)
    grasp = min(range(len(hero.gt)), key=lambda i: abs(hero.gt[i].t - 1.9))
    write_png(out / "hero-white.png", 320, 320, hero.gt[grasp].rgb)
    write_png(out / "hero-photoreal.png", 320, 320, hero.photoreal.frames[grasp].rgb)
    raw_depth = hero.gt[grasp].depth
    far = max(raw_depth)
    valid = [d for d in raw_depth if d < far - 1e-6]
    lo, hi = min(valid), max(valid)
    span = max(hi - lo, 1e-6)
    write_png(
        out / "hero-depth.png",
        320,
        320,
        bytes(
            18 if d >= far - 1e-6 else int(245 - ((d - lo) / span) * 205)
            for d in raw_depth
        ),
        channels=1,
    )

    print("[4/6] batch grid @ 96px")
    batch = batch_data(out)

    print("[5/6] real CLI transcripts")
    cli = cli_transcripts()

    print("[6/6] metrics + morphology cards")
    payload = {
        "clean": metrics_of(clean, args.size, 6, 0.0),
        "drifted": metrics_of(drifted, args.size, 6, 25.0),
        "morphologies": morphology_frames(out),
        "episode": episode_data(clean),
        "batch": batch,
        "cli": cli,
        "sheets": {
            "white": {"src": "assets/white-sheet.png", "cols": SHEET_COLS, "size": args.size, "frames": len(clean.gt)},
            "photoreal": {"src": "assets/photoreal-sheet.png", "cols": SHEET_COLS, "size": args.size, "frames": len(clean.photoreal.frames)},
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Also emit a script form so the page works when opened straight from disk
    # (file:// blocks fetch of metrics.json in most browsers).
    (out / "metrics.js").write_text(
        "window.EMBODIGEN_METRICS = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print("wrote", out / "metrics.json")
    print(f"clean p95 {payload['clean']['p95_mm']} mm / drifted p95 {payload['drifted']['p95_mm']} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
