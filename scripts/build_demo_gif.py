"""Render the README demo GIF straight from the pipeline.

No ffmpeg, no Pillow: the package's own GIF89a encoder (``embodigen.gifio``)
is enough for a side-by-side white-model / photoreal loop, and it keeps the
asset reproducible with the same standard-library-only promise as the core.

    python scripts/build_demo_gif.py --out docs/assets/demo.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodigen.gifio import hstack, write_gif  # noqa: E402
from embodigen.pipeline import Pipeline, PipelineConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/assets/demo.gif")
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args()

    pipeline = Pipeline(
        PipelineConfig(
            out_dir=Path("out/demo"),
            resolution=args.size,
            fps=args.fps,
            max_attempts=1,
            inject_drift_mm=0.0,
            grain=0.0,
            pack=False,
        )
    )
    episode = pipeline.run("把红色方块放进盒子", seed=0)

    count = min(len(episode.gt), args.max_frames)
    frames = []
    for i in range(count):
        composed, width, height = hstack(
            [episode.gt[i].rgb, episode.photoreal.frames[i].rgb], args.size
        )
        frames.append(composed)

    delay = max(2, int(round(100.0 / args.fps)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(out, frames, width, height, delay)
    print(f"wrote {out} — {count} frames, {width}x{height}, {out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
