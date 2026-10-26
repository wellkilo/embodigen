"""VPT contractor recordings (Minecraft) -> raw recordings.

Each VPT recording is a ``<name>.mp4`` at 20 fps plus a ``<name>.jsonl`` with
one line per frame::

    {"mouse": {"dx": .., "dy": .., "buttons": [0], "newButtons": [], "dwheel": 0, ...},
     "keyboard": {"keys": ["key.keyboard.w"], ...},
     "isGuiOpen": false, "hotbar": 0, ...}

(checked against the public contractor index ``all_6xx_Jun_29.json``; those
logs also carry ``yaw``/``pitch``/``xpos``/``ypos``/``zpos``, kept as the pose).

Conversion keeps the video where it is (a ``video.mp4`` symlink in the
recording directory, frames addressed as ``video.mp4#t=<seconds>``) and turns
the per-frame key and button sets into press/release events. Two VPT quirks
are handled the way the VPT loader handles them:

* an attack button reported as held on the first frame is a capture artefact
  and is ignored until it is released;
* the camera reports relative ``dx``/``dy`` directly, so absolute cursor
  positions are not logged (they are only meaningful while a GUI is open and
  would trip the recentering filter otherwise).

The player name in the file stem (``cheeky-cornflower-setter-<hash>-<date>``)
is the stream used for per-stream caps.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from embodigen.gameplay.actions import canonical_key
from embodigen.gameplay.recording import write_recording
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.types import (
    BUTTON,
    KEY,
    MOVE,
    WHEEL,
    InputEvent,
    Observation,
    Recording,
    RecordingMeta,
)

VPT_FPS = 20.0
VPT_SIZE = (640, 360)
_BUTTONS = {0: "mouse_left", 1: "mouse_right", 2: "mouse_middle"}
_STEM = re.compile(r"^(?P<player>.+?)-[0-9a-f]{12}-\d{8}-\d{6}$")
_SPLIT_DIRS = {"test": "val", "val": "val", "validation": "val", "heldout": "val"}


def stream_of(stem: str) -> str:
    match = _STEM.match(stem)
    return f"vpt:{match.group('player')}" if match else f"vpt:{stem}"


class VPTSource:
    name = "vpt"

    def __init__(self, world: str = "minecraft", fps: float = VPT_FPS, size: tuple[int, int] = VPT_SIZE) -> None:
        self.world, self.fps, self.size = world, fps, size

    def discover(self, root: Path) -> list[Path]:
        return sorted(Path(root).rglob("*.jsonl"))

    def convert(self, path: Path, out_root: Path, registry: WorldRegistry) -> RecordingMeta:
        path = Path(path)
        steps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rec_id = f"vpt-{path.stem}"
        root = Path(out_root) / rec_id
        root.mkdir(parents=True, exist_ok=True)

        video = path.with_suffix(".mp4")
        frame_name = "video.mp4"
        notes = [f"source={path}"]
        if video.exists():
            link = root / frame_name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(video.resolve(), link)
        else:
            notes.append("video missing: frames are references only")

        events: list[InputEvent] = []
        observations: list[Observation] = []
        held: set[str] = set()
        attack_stuck = False
        for i, step in enumerate(steps):
            t = i / self.fps
            mouse = step.get("mouse", {}) or {}
            keyboard = step.get("keyboard", {}) or {}
            buttons = list(mouse.get("buttons", []) or [])
            if i == 0 and mouse.get("newButtons") == [0]:
                attack_stuck = True
            elif attack_stuck and 0 in (mouse.get("newButtons") or []):
                attack_stuck = False
            if attack_stuck:
                buttons = [b for b in buttons if b != 0]

            now = {canonical_key(k) for k in keyboard.get("keys", []) or []}
            now |= {_BUTTONS[b] for b in buttons if b in _BUTTONS}
            for code in sorted(held - now):
                events.append(InputEvent(t=t, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=False))
            for code in sorted(now - held):
                events.append(InputEvent(t=t, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=True))
            held = now

            dx, dy = float(mouse.get("dx", 0) or 0), float(mouse.get("dy", 0) or 0)
            if dx or dy:
                events.append(InputEvent(t=t, kind=MOVE, dx=dx, dy=dy))
            dz = float(mouse.get("dwheel", 0) or 0)
            if dz:
                events.append(InputEvent(t=t, kind=WHEEL, dz=max(-1.0, min(1.0, dz))))

            state = {"gui_open": bool(step.get("isGuiOpen", False))}
            if "hotbar" in step:
                state["hotbar"] = step["hotbar"]
            # 6.x/7.x/8.x/9.x/10.x contractor logs also carry the player pose.
            if "yaw" in step:
                state["pose"] = [step.get(k) for k in ("xpos", "ypos", "zpos", "yaw", "pitch")]
            observations.append(Observation(t=t, index=i, frame=f"{frame_name}#t={t:.3f}", state=state))

        split = next((_SPLIT_DIRS[p.name.lower()] for p in path.parents if p.name.lower() in _SPLIT_DIRS), "")
        meta = RecordingMeta(
            recording_id=rec_id,
            world=self.world,
            source=self.name,
            stream=stream_of(path.stem),
            duration=round(len(steps) / self.fps, 6),
            frame_hz=self.fps,
            width=self.size[0],
            height=self.size[1],
            mouse_scale=registry.get(self.world).mouse_scale(self.name),
            engine="minecraft (human)",
            exact_state=False,
            split_hint=split,
            notes=notes,
        )
        write_recording(Recording(meta=meta, events=events, observations=observations), root)
        return meta
