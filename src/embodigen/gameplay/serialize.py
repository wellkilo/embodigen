"""Windows -> training samples in the GameScaling template, written as shards.

One sample is one window: alternating user/assistant turns, an image per user
turn, and per assistant turn::

    <think>
    [sparse rationale, when this turn has one]
    </think>
    <|action_start|> Δx ; Δy ; Δz ; k1 ; ... ; k6 <|action_end|>

Rows follow the ``messages`` + ``images`` shape used by common multimodal SFT
loaders (one ``<image>`` placeholder per user turn). Loss goes on assistant
text only; image tokens are context. Image paths are relative to the
``media_root`` recorded in the manifest; frames kept inside a source video
are addressed as ``<video>#t=<seconds>``.

Shards are assigned by hash of the window id, so writing is streaming (one
recording's turns in memory at a time) and each shard is shuffled on close.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

from embodigen.gameplay.actions import encode_action
from embodigen.gameplay.annotate import RationaleCache, selected
from embodigen.gameplay.types import Turn, Window
from embodigen.gameplay.windows import h01

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
IMAGE_TOKEN = "<image>"


def assistant_text(turn: Turn, rationale: str | None) -> str:
    think = f"{THINK_OPEN}\n{rationale}\n{THINK_CLOSE}" if rationale else f"{THINK_OPEN}\n{THINK_CLOSE}"
    return f"{think}\n{encode_action(turn.action)}"


def build_sample(
    window: Window,
    turns: list[Turn],
    media_prefix: str,
    annotator: Any,
    cache: RationaleCache,
    rate: float,
    salt: str,
    frame_root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Returns (sample, number of rationales in it). ``frame_root`` is the
    recording's raw directory, for annotators that look at the frame."""
    clip = turns[window.start : window.start + window.turns]
    if len(clip) != window.turns:
        raise ValueError(f"{window.window_id}: expected {window.turns} turns, got {len(clip)}")
    messages: list[dict[str, str]] = []
    images: list[str] = []
    thinks = 0
    for turn in clip:
        rationale = None
        if rate > 0 and selected(window.recording_id, turn.index, rate, salt):
            context = turns[max(0, turn.index - 10) : turn.index]
            media = (frame_root, turn.frame) if frame_root is not None else None
            rationale = cache.get_or_make(annotator, window.recording_id, turn, context, window.world, media)
            thinks += 1
        messages.append({"role": "user", "content": IMAGE_TOKEN})
        messages.append({"role": "assistant", "content": assistant_text(turn, rationale)})
        images.append(f"{media_prefix}/{turn.frame}")
    sample = {
        "id": window.window_id,
        "world": window.world,
        "recording": window.recording_id,
        "start_turn": window.start,
        "t0": clip[0].t,
        "messages": messages,
        "images": images,
    }
    return sample, thinks


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, expected: int, shard_size: int, seed: int) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.n = max(1, math.ceil(expected / max(1, shard_size)))
        self.seed = seed
        self._fh: dict[int, Any] = {}
        self.counts = [0] * self.n

    def path(self, i: int) -> Path:
        return self.out_dir / f"{self.prefix}-{i:05d}-of-{self.n:05d}.jsonl"

    def add(self, sample: dict[str, Any]) -> None:
        i = min(self.n - 1, int(h01("shard", self.seed, sample["id"]) * self.n))
        if i not in self._fh:
            self._fh[i] = self.path(i).open("w", encoding="utf-8")
        self._fh[i].write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.counts[i] += 1

    def close(self) -> list[dict[str, Any]]:
        for fh in self._fh.values():
            fh.close()
        shards = []
        for i in range(self.n):
            path = self.path(i)
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            random.Random(f"{self.seed}:{self.prefix}:{i}").shuffle(lines)
            blob = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
            path.write_bytes(blob)
            shards.append({"path": path.name, "samples": len(lines), "sha256": hashlib.sha256(blob).hexdigest()})
        return shards


def write_samples(
    windows: list[Window],
    load_turns: Callable[[str], list[Turn]],
    media_prefix: Callable[[str], str],
    writer: ShardWriter,
    annotator: Any,
    cache: RationaleCache,
    rate: float,
    salt: str,
    frame_root: Callable[[str], Path] | None = None,
) -> dict[str, Any]:
    """Stream ``windows`` (grouped by recording) into ``writer``; returns reasoning stats."""
    turns_total = thinks_total = samples_with = 0
    current, turns = None, []
    for window in sorted(windows, key=lambda w: (w.recording_id, w.start)):
        if window.recording_id != current:
            current, turns = window.recording_id, load_turns(window.recording_id)
        root = frame_root(current) if frame_root else None
        sample, thinks = build_sample(window, turns, media_prefix(current), annotator, cache, rate, salt, root)
        writer.add(sample)
        turns_total += window.turns
        thinks_total += thinks
        samples_with += thinks > 0
    n = len(windows)
    return {
        "assistant_turns": turns_total,
        "rationales": thinks_total,
        "density": round(thinks_total / turns_total, 5) if turns_total else 0.0,
        "samples_with_rationale": round(samples_with / n, 4) if n else 0.0,
        "rationales_per_sample": round(thinks_total / n, 4) if n else 0.0,
    }
