"""Turn a packed episode into something a human can look at.

``embodigen preview`` lays the four channels of every frame side by side --
white model, photoreal, depth, segmentation -- and writes an animated GIF.
It reads only the episode directory, so it doubles as a check that the
packed artefacts are complete and consistent.
"""

from __future__ import annotations

from pathlib import Path

from embodigen.gifio import hstack, write_gif
from embodigen.modules.pack import read_episode

CHANNELS = ("white", "photoreal", "depth", "seg")

# Perceptual-ish ramp for depth: near is bright and warm, far is dark and cool.
_DEPTH_STOPS = (
    (0.00, (252, 238, 170)),
    (0.30, (247, 152, 76)),
    (0.60, (196, 62, 104)),
    (0.85, (88, 40, 132)),
    (1.00, (24, 22, 60)),
)

# Instance 1 is the support surface in every template, so it gets a quiet
# colour and the objects that matter carry the saturated ones.
_SEG_COLORS = (
    (46, 52, 68),
    (230, 96, 82),
    (86, 180, 233),
    (240, 200, 80),
    (120, 200, 140),
    (190, 120, 220),
    (240, 150, 70),
    (90, 210, 210),
    (220, 110, 170),
    (160, 170, 90),
    (110, 130, 230),
)
_BACKGROUND = (14, 16, 22)


def _ramp(x: float) -> tuple[int, int, int]:
    x = min(1.0, max(0.0, x))
    for (x0, c0), (x1, c1) in zip(_DEPTH_STOPS, _DEPTH_STOPS[1:]):
        if x <= x1:
            s = (x - x0) / max(x1 - x0, 1e-9)
            return (
                int(c0[0] + (c1[0] - c0[0]) * s),
                int(c0[1] + (c1[1] - c0[1]) * s),
                int(c0[2] + (c1[2] - c0[2]) * s),
            )
    return _DEPTH_STOPS[-1][1]


def depth_to_rgb(depth: list[float], near: float, far: float) -> bytes:
    """Colourise depth on a fixed range so frames are comparable.

    The ramp runs over inverse depth (disparity): a long table receding to
    the horizon would otherwise swallow the whole range, and the objects in
    the foreground -- the ones the labels are about -- would all look alike.
    """
    inv_near, inv_far = 1.0 / max(near, 1e-6), 1.0 / max(far, 1e-6)
    span = max(inv_near - inv_far, 1e-9)
    out = bytearray(len(depth) * 3)
    for p, d in enumerate(depth):
        color = _BACKGROUND if d >= far else _ramp((inv_near - 1.0 / max(d, 1e-6)) / span)
        out[p * 3 : p * 3 + 3] = bytes(color)
    return bytes(out)


def seg_to_rgb(seg: list[int]) -> bytes:
    out = bytearray(len(seg) * 3)
    for p, sid in enumerate(seg):
        color = _BACKGROUND if sid == 0 else _SEG_COLORS[(sid - 1) % len(_SEG_COLORS)]
        out[p * 3 : p * 3 + 3] = bytes(color)
    return bytes(out)


def upscale(rgb: bytes, size: int, factor: int) -> bytes:
    """Nearest-neighbour upscale: keeps label pixels crisp instead of blurred."""
    if factor <= 1:
        return rgb
    big = size * factor
    out = bytearray(big * big * 3)
    for y in range(size):
        row = bytearray()
        for x in range(size):
            row += rgb[(y * size + x) * 3 : (y * size + x) * 3 + 3] * factor
        for k in range(factor):
            start = ((y * factor + k) * big) * 3
            out[start : start + big * 3] = row
    return bytes(out)


def render_preview(
    episode_root: str | Path,
    out: str | Path,
    channels: tuple[str, ...] = CHANNELS,
    fps: float = 8.0,
    scale: int = 0,
    max_frames: int = 0,
) -> tuple[Path, int, int, int]:
    """Write a GIF preview; returns (path, frames, width, height)."""
    unknown = [c for c in channels if c not in CHANNELS]
    if unknown:
        raise SystemExit(f"unknown channel(s) {', '.join(unknown)}; choose from {', '.join(CHANNELS)}")

    scene, gt, rendered = read_episode(episode_root)
    size = scene.camera.width
    factor = scale or max(1, round(160 / size))
    far = scene.camera.far
    valid = [d for frame in gt for d in frame.depth if d < far]
    near_d, far_d = (min(valid), max(valid)) if valid else (0.0, 1.0)

    count = len(gt) if max_frames <= 0 else min(len(gt), max_frames)
    frames: list[bytes] = []
    width = height = 0
    for i in range(count):
        panels = []
        for channel in channels:
            if channel == "white":
                rgb = gt[i].rgb
            elif channel == "photoreal":
                rgb = rendered.frames[i].rgb
            elif channel == "depth":
                rgb = depth_to_rgb(gt[i].depth, near_d, far_d + 1e-6)
            else:
                rgb = seg_to_rgb(gt[i].seg)
            panels.append(upscale(rgb, size, factor))
        composed, width, height = hstack(panels, size * factor, gap=max(2, factor))
        frames.append(composed)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(out, frames, width, height, max(2, int(round(100.0 / fps))))
    return out, count, width, height
