"""The canonical keyboard-mouse action and its text form.

GameScaling feeds every world through one interface: observations at 5 Hz,
and for each one a 200 ms action packet made of a mouse displacement
(Δx, Δy in [-1000, 1000]), a wheel step (Δz in [-5, 5]) and six 33 ms key
slots holding 0-4 keys or mouse buttons each. This module owns that format:
resampling raw input events into packets, and the text the model is trained
to emit::

    <|action_start|> 12 ; -3 ; 0 ; w ; w ; w,space ; w ; w ; w <|action_end|>

The paper does not publish its key spelling, so the slot syntax (``,``
between keys, ``none`` for an empty slot) lives here and nowhere else.
"""

from __future__ import annotations

import bisect
import re
from typing import Iterable

from embodigen.gameplay.types import (
    BUTTON,
    KEY,
    MOVE,
    WHEEL,
    ActionChunk,
    InputEvent,
    Observation,
    Turn,
)

OBS_HZ = 5.0
CHUNK_S = 1.0 / OBS_HZ  # 200 ms
SUBSTEPS = 6  # 33 ms each, i.e. 30 Hz control
MOUSE_LIMIT = 1000
WHEEL_LIMIT = 5
KEYS_PER_SLOT = 4

ACTION_START, ACTION_END = "<|action_start|>", "<|action_end|>"
EMPTY_SLOT = "none"

# Canonical key names, in the order kept when a slot overflows: movement and
# mouse buttons first, because they carry most of the control signal.
KEY_ORDER: tuple[str, ...] = (
    "w", "a", "s", "d", "mouse_left", "mouse_right", "space", "shift", "ctrl",
    "e", "q", "r", "f", "c", "x", "z", "v", "g", "t", "tab", "esc",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "up", "down", "left", "right", "enter", "alt", "mouse_middle",
    "b", "h", "i", "j", "k", "l", "m", "n", "o", "p", "u", "y",
    "f1", "f2", "f3", "f4", "f5", "backspace", "capslock",
)
_RANK = {k: i for i, k in enumerate(KEY_ORDER)}
_ALIASES = {
    "escape": "esc", "spacebar": "space", "return": "enter",
    "lshift": "shift", "rshift": "shift", "left.shift": "shift", "right.shift": "shift",
    "left_shift": "shift", "right_shift": "shift",
    "lctrl": "ctrl", "rctrl": "ctrl", "control": "ctrl", "left.control": "ctrl",
    "right.control": "ctrl", "left_control": "ctrl", "right_control": "ctrl",
    "lalt": "alt", "ralt": "alt", "left.alt": "alt", "right.alt": "alt",
    "lmb": "mouse_left", "rmb": "mouse_right", "mmb": "mouse_middle",
    "mouse1": "mouse_left", "mouse2": "mouse_right", "mouse3": "mouse_middle",
    "caps.lock": "capslock",
    # Open-P2P spells keys after their physical names.
    "leftshift": "shift", "rightshift": "shift", "leftcontrol": "ctrl", "rightcontrol": "ctrl",
    "leftalt": "alt", "rightalt": "alt", "uparrow": "up", "downarrow": "down",
    "leftarrow": "left", "rightarrow": "right", "backquote": "`", "graveaccent": "`",
    # "," and ";" are separators in the action text.
    ",": "comma", ";": "semicolon",
}


def canonical_key(name: str) -> str:
    """Map a source's key spelling onto the canonical vocabulary."""
    key = name.strip().lower()
    for prefix in ("key.keyboard.", "key.mouse.", "key_", "vk_"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return _ALIASES.get(key, key)


def key_rank(key: str) -> tuple[int, str]:
    return (_RANK.get(key, len(_RANK)), key)


def _clip(value: float, limit: int) -> int:
    return max(-limit, min(limit, int(round(value))))


# -- text codec ---------------------------------------------------------------

def encode_action(chunk: ActionChunk) -> str:
    slots = [",".join(keys) if keys else EMPTY_SLOT for keys in chunk.keys]
    body = " ; ".join([str(chunk.dx), str(chunk.dy), str(chunk.dz), *slots])
    return f"{ACTION_START} {body} {ACTION_END}"


_ACTION_RE = re.compile(re.escape(ACTION_START) + r"(.*?)" + re.escape(ACTION_END), re.S)


def decode_action(text: str) -> ActionChunk:
    """Parse the model's action text back; raises ``ValueError`` on bad output."""
    match = _ACTION_RE.search(text)
    if not match:
        raise ValueError("no action span")
    parts = [p.strip() for p in match.group(1).split(";")]
    if len(parts) != 3 + SUBSTEPS:
        raise ValueError(f"expected {3 + SUBSTEPS} fields, got {len(parts)}")
    dx, dy, dz = (int(p) for p in parts[:3])
    keys = tuple(
        () if p in ("", EMPTY_SLOT) else tuple(k.strip() for k in p.split(",") if k.strip())
        for p in parts[3:]
    )
    return ActionChunk(
        dx=_clip(dx, MOUSE_LIMIT), dy=_clip(dy, MOUSE_LIMIT), dz=_clip(dz, WHEEL_LIMIT), keys=keys
    )


# -- resampling ---------------------------------------------------------------

def held_intervals(events: Iterable[InputEvent], end: float) -> dict[str, list[tuple[float, float]]]:
    """Press/release pairs per key; a key still down at the end is closed there."""
    open_at: dict[str, float] = {}
    spans: dict[str, list[tuple[float, float]]] = {}
    for ev in events:
        if ev.kind not in (KEY, BUTTON):
            continue
        if ev.down:
            open_at.setdefault(ev.code, ev.t)
        elif ev.code in open_at:
            start = open_at.pop(ev.code)
            # A press and release logged on the same tick is still a tap.
            spans.setdefault(ev.code, []).append((start, max(ev.t, start + 1e-3)))
    for code, start in open_at.items():
        spans.setdefault(code, []).append((start, end))
    return spans


class Canonicalizer:
    """Raw events + frames -> 5 Hz turns of canonical action packets.

    A key counts as held in a sub-step if it was down at any instant of it, so
    a tap shorter than 33 ms still shows up. Mouse and wheel motion is summed
    over the whole packet, rescaled by the source's ``mouse_scale`` and
    clipped to the canonical range.
    """

    name = "canonical-kbm-v1"

    def __init__(self, obs_hz: float = OBS_HZ, substeps: int = SUBSTEPS) -> None:
        self.chunk = 1.0 / obs_hz
        self.substeps = substeps

    def turns(
        self,
        events: list[InputEvent],
        observations: list[Observation],
        duration: float,
        mouse_scale: float = 1.0,
    ) -> list[Turn]:
        if not observations:
            return []
        events = sorted(events, key=lambda e: e.t)
        spans = held_intervals(events, duration)
        # Sweep the press intervals forward once instead of rescanning them
        # for every packet: hours of input stay linear.
        intervals = sorted((a, b, code) for code, ivs in spans.items() for a, b in ivs)
        next_iv = 0
        active: list[tuple[float, float, str]] = []
        motion = [e for e in events if e.kind in (MOVE, WHEEL)]
        motion_t = [e.t for e in motion]
        obs_t = [o.t for o in observations]

        out: list[Turn] = []
        n = int((duration + 1e-9) // self.chunk)
        for i in range(n):
            t0 = i * self.chunk
            t1 = t0 + self.chunk
            if t1 > duration + 1e-9:
                break
            # The newest frame captured at or before the packet starts.
            j = bisect.bisect_right(obs_t, t0 + 1e-6) - 1
            if j < 0:
                continue
            # Logs are microsecond-rounded; don't let float noise move an
            # event logged exactly on a boundary into the previous packet.
            lo = bisect.bisect_left(motion_t, t0 - 1e-6)
            hi = bisect.bisect_left(motion_t, t1 - 1e-6)
            dx = sum(e.dx for e in motion[lo:hi] if e.kind == MOVE) * mouse_scale
            dy = sum(e.dy for e in motion[lo:hi] if e.kind == MOVE) * mouse_scale
            dz = sum(e.dz for e in motion[lo:hi] if e.kind == WHEEL)

            while next_iv < len(intervals) and intervals[next_iv][0] < t1 - 1e-6:
                active.append(intervals[next_iv])
                next_iv += 1
            active = [iv for iv in active if iv[1] > t0]

            slots: list[tuple[str, ...]] = []
            truncated = 0
            step = self.chunk / self.substeps
            for s in range(self.substeps):
                a, b = t0 + s * step, t0 + (s + 1) * step
                held = sorted({code for p, r, code in active if p < b - 1e-6 and r > a + 1e-6}, key=key_rank)
                truncated += max(0, len(held) - KEYS_PER_SLOT)
                slots.append(tuple(held[:KEYS_PER_SLOT]))

            obs = observations[j]
            out.append(
                Turn(
                    index=len(out),
                    t=round(t0, 6),
                    frame=obs.frame,
                    action=ActionChunk(
                        dx=_clip(dx, MOUSE_LIMIT),
                        dy=_clip(dy, MOUSE_LIMIT),
                        dz=_clip(dz, WHEEL_LIMIT),
                        keys=tuple(slots),
                    ),
                    state=obs.state,
                    truncated_keys=truncated,
                )
            )
        return out


def turn_to_json(turn: Turn) -> dict:
    row = {
        "index": turn.index,
        "t": turn.t,
        "frame": turn.frame,
        "dx": turn.action.dx,
        "dy": turn.action.dy,
        "dz": turn.action.dz,
        "keys": [list(k) for k in turn.action.keys],
        "idle": turn.action.idle,
    }
    if turn.state is not None:
        row["state"] = turn.state
    if turn.truncated_keys:
        row["truncated_keys"] = turn.truncated_keys
    return row


def turn_from_json(row: dict) -> Turn:
    return Turn(
        index=row["index"],
        t=row["t"],
        frame=row["frame"],
        action=ActionChunk(
            dx=row["dx"], dy=row["dy"], dz=row["dz"], keys=tuple(tuple(k) for k in row["keys"])
        ),
        state=row.get("state"),
        truncated_keys=row.get("truncated_keys", 0),
    )
