"""Turn a recording into something a human can look at.

One GIF frame per 5 Hz turn: the observation, and when the engine exported
them its depth and instance labels, with an input strip underneath -- a lit
box for each held key and a bar for the mouse ``dx`` of the packet that
follows the frame. Watching the bar lead or trail the view is the quickest
way to eyeball an action/frame desync.
"""

from __future__ import annotations

import struct
from pathlib import Path

from embodigen.gameplay.actions import Canonicalizer
from embodigen.gameplay.recording import read_recording
from embodigen.gifio import write_gif
from embodigen.pngio import read_png
from embodigen.preview import depth_to_rgb, seg_to_rgb

HUD_KEYS = ("w", "a", "s", "d", "space", "shift", "mouse_left", "mouse_right")
_BG = (10, 12, 16)
_OFF = (44, 50, 62)
_ON = (82, 196, 120)
_BAR = (90, 160, 250)


def _scale(rgb: bytes, w: int, h: int, k: int) -> bytes:
    if k <= 1:
        return rgb
    out = bytearray(w * k * h * k * 3)
    for y in range(h):
        row = bytearray()
        for x in range(w):
            row += rgb[(y * w + x) * 3 : (y * w + x) * 3 + 3] * k
        for j in range(k):
            start = (y * k + j) * w * k * 3
            out[start : start + w * k * 3] = row
    return bytes(out)


def _blit(canvas: bytearray, cw: int, src: bytes, w: int, h: int, x0: int, y0: int) -> None:
    for y in range(h):
        dst = ((y0 + y) * cw + x0) * 3
        canvas[dst : dst + w * 3] = src[y * w * 3 : (y + 1) * w * 3]


def _rect(canvas: bytearray, cw: int, x0: int, y0: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    line = bytes(color) * w
    for y in range(y0, y0 + h):
        dst = (y * cw + x0) * 3
        canvas[dst : dst + w * 3] = line


def render_recording(root: Path, out: Path, max_turns: int = 0, scale: int = 0, fps: float = 5.0) -> tuple[Path, int, int, int]:
    root = Path(root)
    rec = read_recording(root)
    turns = Canonicalizer().turns(rec.events, rec.observations, rec.meta.duration, rec.meta.mouse_scale)
    if not turns or not turns[0].frame.endswith(".png"):
        raise SystemExit(f"{root}: preview needs decoded PNG frames")
    by_frame = {o.frame: o for o in rec.observations}
    w, h, _, _ = read_png(root / turns[0].frame)
    k = scale or max(1, round(240 / w))
    pw, ph = w * k, h * k
    first = by_frame[turns[0].frame]
    channels = 1 + (first.depth is not None) + (first.labels is not None)
    gap = max(2, k)
    hud_h = 10 * k
    cw = pw * channels + gap * (channels - 1)
    ch = ph + gap + hud_h

    count = len(turns) if max_turns <= 0 else min(len(turns), max_turns)
    frames: list[bytes] = []
    for turn in turns[:count]:
        canvas = bytearray(bytes(_BG) * (cw * ch))
        fw, fh, c, px = read_png(root / turn.frame)
        if c != 3:
            px = bytes(b for i in range(fw * fh) for b in (px[i * c : i * c + 3] if c >= 3 else px[i : i + 1] * 3))
        panels = [px]
        obs = by_frame[turn.frame]
        if obs.depth:
            depth = list(struct.unpack(f"<{w * h}f", (root / obs.depth).read_bytes()))
            valid = [d for d in depth if d > 0]
            panels.append(depth_to_rgb(depth, min(valid, default=0.1), max(valid, default=1.0) + 1e-6))
        if obs.labels:
            panels.append(seg_to_rgb(list(struct.unpack(f"<{w * h}H", (root / obs.labels).read_bytes()))))
        for i, panel in enumerate(panels[:channels]):
            _blit(canvas, cw, _scale(panel, w, h, k), pw, ph, i * (pw + gap), 0)

        held = set().union(*turn.action.keys)
        box = 8 * k
        y0 = ph + gap + k
        for i, key in enumerate(HUD_KEYS):
            _rect(canvas, cw, k + i * (box + k), y0, box, box, _ON if key in held else _OFF)
        bar_x0 = k + len(HUD_KEYS) * (box + k) + 2 * k
        bar_w = cw - bar_x0 - k
        if bar_w > 4:
            mid = bar_x0 + bar_w // 2
            _rect(canvas, cw, bar_x0, y0 + box // 2 - k // 2 - 1, bar_w, max(1, k), _OFF)
            span = int(abs(turn.action.dx) / 1000.0 * (bar_w // 2))
            if span:
                _rect(canvas, cw, mid if turn.action.dx > 0 else mid - span, y0, span, box, _BAR)
            _rect(canvas, cw, mid, y0, max(1, k // 2), box, _ON)
        frames.append(bytes(canvas))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(out, frames, cw, ch, max(2, int(round(100.0 / fps))))
    return out, count, cw, ch
