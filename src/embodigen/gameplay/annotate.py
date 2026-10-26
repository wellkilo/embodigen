"""Sparse reasoning annotations.

GameScaling attaches a short retrospective rationale to roughly 5.7% of
assistant turns and leaves the rest with an empty ``<think>`` span. Two rules
keep that from confounding a scaling study:

* **Selection is a property of the turn, not of the mixture.** A turn is
  annotated iff ``h01(salt, recording, turn) < rate``, so every mixture and
  every budget sees the same density, and a turn shared by two overlapping
  windows carries the same rationale in both.
* **A rationale never edits the label.** It explains the recorded action; the
  action text is produced from the events alone.

Backends implement :class:`embodigen.gameplay.base.Annotator`. ``mock`` writes
templated rationales from the executed action and, for engine rollouts, the
exact game state; a model-backed annotator plugs in through the same
interface and inherits the cache below (``llm`` in :mod:`embodigen.gameplay.llm`
calls a cloud chat API and fills the cache concurrently before a build writes).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from embodigen.gameplay.types import Turn
from embodigen.gameplay.windows import h01

DEFAULT_RATE = 0.057  # middle of the paper's 5.4-6.0% of assistant turns


def selected(recording_id: str, turn_index: int, rate: float, salt: str = "think") -> bool:
    return h01(salt, recording_id, turn_index) < rate


def _motion(turn: Turn) -> str:
    # How many of the six 33 ms slots each key was held in: a packet that
    # taps ``a`` once while holding ``d`` is a strafe right.
    held = {k: sum(k in slot for slot in turn.action.keys) for k in set().union(*turn.action.keys)}
    parts = []
    if held.get("w", 0) > held.get("s", 0):
        parts.append("pushing forward" + (" at a run" if held.get("shift") else ""))
    elif held.get("s", 0) > held.get("w", 0):
        parts.append("backing off")
    if held.get("a", 0) != held.get("d", 0):
        parts.append("strafing " + ("left" if held.get("a", 0) > held.get("d", 0) else "right"))
    if abs(turn.action.dx) > 40:
        parts.append("swinging the view " + ("right" if turn.action.dx > 0 else "left"))
    elif abs(turn.action.dx) > 10:
        parts.append("easing the view " + ("right" if turn.action.dx > 0 else "left"))
    if held.get("mouse_left"):
        parts.append("firing")
    return ", ".join(parts) or "holding still"


class MockAnnotator:
    """Templated rationales; says so in its name so nobody mistakes it for a model."""

    name = "mock-rationale-v0"
    prompt_version = "v2"

    def rationale(self, turn: Turn, context: list[Turn], world: str, media: Any = None) -> str:
        motion = _motion(turn)
        state = turn.state or {}
        visible = state.get("visible") or []
        if visible:
            v = visible[0]
            side = "ahead" if abs(v["bearing"]) < 5 else ("to my right" if v["bearing"] > 0 else "to my left")
            article = "an" if v["kind"][0] in "aeiou" else "a"
            return (
                f"There is {article} {v['kind']} {side}, about {v['distance']:.1f} m away, "
                f"so I keep it in view: {motion}."
            )
        if state.get("gui_open"):
            return f"An inventory screen is open, so I manage items: {motion}."
        recent = [t for t in context[-5:] if not t.action.idle]
        if not recent:
            return f"Nothing needs attention right now; {motion}."
        return f"Nothing worth chasing is in sight, so I keep exploring: {motion}."


class RationaleCache:
    """Append-only JSONL keyed by (annotator, prompt version, recording, turn).

    Paying for a rationale once matters with a model backend; re-running a
    build after a mixture tweak should cost nothing.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self.store: dict[str, str] = {}
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.store[row["key"]] = row["text"]
        self._fh = None

    @staticmethod
    def key(annotator: str, version: str, recording_id: str, turn_index: int) -> str:
        raw = f"{annotator}\x1f{version}\x1f{recording_id}\x1f{turn_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get_or_make(self, annotator, recording_id: str, turn: Turn, context: list[Turn], world: str,
                    media: Any = None) -> str:
        k = self.key(annotator.name, annotator.prompt_version, recording_id, turn.index)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        self.misses += 1
        text = annotator.rationale(turn, context, world, media).strip()
        self._put(k, recording_id, turn.index, text)
        return text

    def prefill(self, annotator, jobs: list[tuple[str, Turn, list[Turn], str, Any]], workers: int) -> int:
        """Make every missing rationale in ``jobs`` with ``workers`` concurrent
        calls, writing each to the cache as it lands. A failure stops the run
        but keeps what was already paid for. Returns how many were made."""
        todo: dict[str, tuple[str, Turn, list[Turn], str, Any]] = {}
        for job in jobs:
            k = self.key(annotator.name, annotator.prompt_version, job[0], job[1].index)
            if k not in self.store:
                todo.setdefault(k, job)
        if not todo:
            return 0
        made, error = 0, None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(annotator.rationale, job[1], job[2], job[3], job[4]): (k, job)
                       for k, job in todo.items()}
            try:
                for fut in as_completed(futures):
                    if fut.cancelled():
                        continue
                    k, job = futures[fut]
                    try:
                        text = fut.result().strip()
                    except Exception as exc:  # stop launching, but keep harvesting what was paid for
                        if error is None:
                            error = exc
                            for other in futures:
                                other.cancel()
                        continue
                    self._put(k, job[0], job[1].index, text)
                    made += 1
            except BaseException:  # Ctrl-C: cancel the rest, keep the file consistent
                for fut in futures:
                    fut.cancel()
                self.close()
                raise
        if error is not None:
            self.close()
            raise error
        return made

    def _put(self, k: str, recording_id: str, turn_index: int, text: str) -> None:
        self.store[k] = text
        if self.path:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(json.dumps({"key": k, "recording": recording_id, "turn": turn_index, "text": text},
                                      ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def make_annotator(name: str, reasoning: dict[str, Any]):
    """Instantiate a registered annotator; backends with settings read them from the study."""
    from embodigen.modules.base import resolve

    cls = resolve("annotate", name)
    return cls.from_config(reasoning) if hasattr(cls, "from_config") else cls()
