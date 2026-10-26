"""Raw recordings on disk.

One directory per recording, readable without this package::

    raw/<recording_id>/
    ├── recording.json   # RecordingMeta
    ├── events.jsonl     # one InputEvent per line, time-ordered
    ├── obs.jsonl        # one Observation per line
    ├── frames/          # RGB PNGs (engine rollouts) -- or a video the obs rows point into
    ├── depth/           # float32 planar depth, engine truth (optional)
    └── labels/          # uint16 instance ids, engine truth (optional)

Human recordings and engine rollouts share this layout, so everything
downstream of ingestion is source-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from embodigen.gameplay.types import InputEvent, Observation, Recording, RecordingMeta


def write_recording(recording: Recording, root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "recording.json").write_text(
        json.dumps(recording.meta.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "events.jsonl").open("w", encoding="utf-8") as fh:
        for ev in sorted(recording.events, key=lambda e: e.t):
            fh.write(json.dumps(ev.to_json(), ensure_ascii=False) + "\n")
    with (root / "obs.jsonl").open("w", encoding="utf-8") as fh:
        for obs in recording.observations:
            fh.write(json.dumps(obs.to_json(), ensure_ascii=False) + "\n")
    return root


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_meta(root: Path) -> RecordingMeta:
    path = Path(root) / "recording.json"
    if not path.exists():
        raise SystemExit(f"{root} is not a recording (missing recording.json)")
    return RecordingMeta.from_json(json.loads(path.read_text(encoding="utf-8")))


def read_recording(root: Path) -> Recording:
    root = Path(root)
    return Recording(
        meta=read_meta(root),
        events=[InputEvent.from_json(r) for r in _jsonl(root / "events.jsonl")],
        observations=[Observation.from_json(r) for r in _jsonl(root / "obs.jsonl")],
    )


def discover(raw_root: Path) -> list[Path]:
    """Every recording directory under ``raw_root`` (any depth), sorted."""
    raw_root = Path(raw_root)
    return sorted(p.parent for p in raw_root.rglob("recording.json"))
