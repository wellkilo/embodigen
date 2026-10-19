"""Build the favicon set from one geometric description.

The mark is the same split the showcase page is built around: an outlined
half (the engine's white model) next to a filled half (what the renderer
puts on it), separated by a gap. It is drawn analytically with 3x3
supersampling so the 16px version stays legible.

    python scripts/build_favicon.py --out docs
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodigen.pngio import write_png  # noqa: E402

BG = (10, 12, 16)
INK = (201, 210, 223)
BLUE = (47, 123, 255)

# Normalised geometry (0..1 square).
BG_RADIUS = 0.2
TILE = 0.15
TILE_END = 0.85
SPLIT = 0.5
GAP = 0.015
SS = 3  # supersampling per axis


def _in_round_rect(x: float, y: float, r: float) -> bool:
    if x < 0 or y < 0 or x > 1 or y > 1:
        return False
    cx = min(max(x, r), 1 - r)
    cy = min(max(y, r), 1 - r)
    dx, dy = x - cx, y - cy
    return dx * dx + dy * dy <= r * r


def _sample(x: float, y: float) -> tuple[int, int, int, int]:
    """Colour at a normalised point; alpha is 0 outside the rounded square."""
    if not _in_round_rect(x, y, BG_RADIUS):
        return (0, 0, 0, 0)

    inside_tile = TILE <= x <= TILE_END and TILE <= y <= TILE_END
    left = x < SPLIT - GAP
    right = x > SPLIT + GAP

    if inside_tile and right:
        return (*BLUE, 255)

    if inside_tile and left:
        # The unrendered half: solid but colourless, like the white model.
        return (*INK, 255)

    return (*BG, 255)


def render(size: int) -> bytes:
    out = bytearray(size * size * 4)
    step = 1.0 / (size * SS)
    for py in range(size):
        for px in range(size):
            r = g = b = a = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = (px * SS + sx + 0.5) * step
                    y = (py * SS + sy + 0.5) * step
                    cr, cg, cb, ca = _sample(x, y)
                    r += cr * ca
                    g += cg * ca
                    b += cb * ca
                    a += ca
            n = SS * SS
            if a == 0:
                continue
            i = (py * size + px) * 4
            out[i] = r // a
            out[i + 1] = g // a
            out[i + 2] = b // a
            out[i + 3] = a // n
    return bytes(out)


def ico(pngs: list[bytes], sizes: list[int]) -> bytes:
    """A PNG-payload ICO, which every current browser accepts."""
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b"", b""
    for size, blob in zip(sizes, pngs):
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
        blobs += blob
    return header + entries + blobs


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="6.4" fill="#0a0c10"/>
  <rect x="4.8" y="4.8" width="10.72" height="22.4" fill="#c9d2df"/>
  <rect x="16.48" y="4.8" width="10.72" height="22.4" fill="#2f7bff"/>
</svg>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 48]
    pngs = []
    for size in sizes:
        pixels = render(size)
        path = out / f"_favicon-{size}.png"
        write_png(path, size, size, pixels, channels=4)
        pngs.append(path.read_bytes())
        path.unlink()

    (out / "favicon.ico").write_bytes(ico(pngs, sizes))
    (out / "favicon.svg").write_text(SVG, encoding="utf-8")

    for size, name in ((32, "favicon-32.png"), (180, "apple-touch-icon.png")):
        write_png(out / name, size, size, render(size), channels=4)

    print("wrote favicon.ico, favicon.svg, favicon-32.png, apple-touch-icon.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
