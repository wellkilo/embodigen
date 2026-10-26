"""Turns -> training windows, plus the two pool-shaping rules that act on them.

* Windows are ``turns`` long with a ``stride`` step (GameScaling: 20 turns,
  stride 10, so each window supervises 4.0 s of actions).
* Idle-dominated windows are downsampled, deterministically, so a run is
  reproducible and the same window is kept or dropped in every mixture.
* Each source stream is capped, by whole recordings, so one prolific player
  or channel cannot dominate a world.

All randomness is a hash of stable ids: no RNG state to thread through.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from embodigen.gameplay.types import Window


def h01(*parts: object) -> float:
    """Stable pseudo-random number in [0, 1) from a tuple of ids."""
    digest = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class WindowConfig:
    turns: int = 20
    stride: int = 10
    idle_turn_frac: float = 0.5
    idle_keep: float = 0.2
    stream_cap_hours: float = 0.0  # 0 disables the cap
    # What a data budget counts. "footage": hours of unique gameplay covered
    # by the chosen windows (overlaps counted once). "window": each window
    # costs its full length, so overlapping windows count shared turns twice.
    budget_unit: str = "footage"

    def __post_init__(self) -> None:
        if self.budget_unit not in ("footage", "window"):
            raise SystemExit(f"budget_unit must be 'footage' or 'window', not {self.budget_unit!r}")

    @property
    def seconds(self) -> float:
        return self.turns * 0.2


def footage_seconds(windows: list[Window]) -> float:
    """Unique gameplay covered by a set of windows (overlaps counted once)."""
    covered: dict[str, set[int]] = {}
    for w in windows:
        covered.setdefault(w.recording_id, set()).update(range(w.start, w.start + w.turns))
    return sum(len(turns) for turns in covered.values()) * 0.2


def make_windows(
    recording_id: str, world: str, stream: str, idle: list[bool], cfg: WindowConfig
) -> list[Window]:
    out = []
    for start in range(0, len(idle) - cfg.turns + 1, cfg.stride):
        frac = sum(idle[start : start + cfg.turns]) / cfg.turns
        out.append(
            Window(
                recording_id=recording_id,
                world=world,
                stream=stream,
                start=start,
                turns=cfg.turns,
                idle_frac=round(frac, 4),
            )
        )
    return out


def is_idle(window: Window, cfg: WindowConfig) -> bool:
    return window.idle_frac >= cfg.idle_turn_frac


def downsample_idle(windows: list[Window], cfg: WindowConfig, salt: str = "idle") -> list[Window]:
    return [w for w in windows if not is_idle(w, cfg) or h01(salt, w.window_id) < cfg.idle_keep]


def cap_streams(
    recordings: list[dict], cfg: WindowConfig, salt: str = "cap"
) -> tuple[list[dict], list[dict]]:
    """Whole recordings per stream, in hash order, until the stream's cap.

    ``recordings`` rows need ``recording_id``, ``stream`` and ``duration``.
    Returns (kept, dropped).
    """
    if cfg.stream_cap_hours <= 0:
        return list(recordings), []
    cap = cfg.stream_cap_hours * 3600.0
    by_stream: dict[str, list[dict]] = {}
    for row in recordings:
        by_stream.setdefault(row["stream"], []).append(row)
    kept, dropped = [], []
    for stream in sorted(by_stream):
        used = 0.0
        for row in sorted(by_stream[stream], key=lambda r: h01(salt, r["recording_id"])):
            if used + row["duration"] <= cap:
                kept.append(row)
                used += row["duration"]
            else:
                dropped.append(row)
    return kept, dropped
