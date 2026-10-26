"""D2E desktop gameplay (OWAMcap) -> raw recordings.

D2E (https://huggingface.co/datasets/open-world-agents/D2E-480p) ships one
folder per game, each holding ``<name>.mcap`` + ``<name>.mkv`` pairs. The MCAP
carries JSON messages on these topics (field names as recorded)::

    screen     desktop/ScreenCaptured  {"utc_ns", "shape": [w, h], "media_ref": {"uri": "<name>.mkv", "pts_ns"}}
    keyboard   desktop/KeyboardEvent   {"event_type": "press"|"release", "vk": <Windows virtual-key code>}
    mouse/raw  desktop/RawMouseEvent   {"last_x", "last_y", "us_flags", "button_flags", "button_data"}
    mouse      desktop/MouseEvent      {"event_type": "move"|"click"|"scroll", "x", "y", "button", "pressed", "dx", "dy"}
    window     desktop/WindowInfo      {"title", "rect"}  (about once a second)

Conversion choices:

* Camera motion comes from ``mouse/raw`` (HID counts, before pointer
  acceleration). Moves are summed into 10 ms bins, which divide the 200 ms
  packet grid exactly, so packets see the same totals at a tenth of the log
  size. Absolute cursor positions are only used when a file has no raw topic.
* Windows key auto-repeat logs repeated ``press`` events; only the first one
  of a hold counts.
* The window title that is in front longest is the game; time spent in any
  other window becomes a focus change, which the gate's non-game-input rule
  reads.
* Frames stay in the video: the ``.mkv`` is symlinked as ``video.mkv`` and
  frames are addressed by presentation time (``video.mkv#t=<seconds>``).
* A session the dataset cut into ``_split_00``, ``_split_01``, ... becomes one
  recording per part sharing a ``session``, so the splitter keeps the parts
  on one side.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter
from pathlib import Path

from embodigen.gameplay.recording import write_recording
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.sources.mcap import McapReader
from embodigen.gameplay.types import (
    BUTTON,
    FOCUS,
    KEY,
    MOVE,
    WHEEL,
    InputEvent,
    Observation,
    Recording,
    RecordingMeta,
)

TOPICS = {"screen", "keyboard", "mouse", "mouse/raw", "window"}
MOVE_BIN_S = 0.01
_SPLIT = re.compile(r"_split_\d+$")

# Windows virtual-key codes -> canonical names (actions.KEY_ORDER spelling).
VK: dict[int, str] = {
    0x08: "backspace", 0x09: "tab", 0x0D: "enter", 0x10: "shift", 0x11: "ctrl", 0x12: "alt",
    0x13: "pause", 0x14: "capslock", 0x1B: "esc", 0x20: "space",
    0x21: "pageup", 0x22: "pagedown", 0x23: "end", 0x24: "home",
    0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down", 0x2D: "insert", 0x2E: "delete",
    0x5B: "win", 0x5C: "win", 0x5D: "menu",
    0xA0: "shift", 0xA1: "shift", 0xA2: "ctrl", 0xA3: "ctrl", 0xA4: "alt", 0xA5: "alt",
    0xBA: "semicolon", 0xBB: "equal", 0xBC: "comma", 0xBD: "minus", 0xBE: "period",
    0xBF: "slash", 0xC0: "`", 0xDB: "lbracket", 0xDC: "backslash", 0xDD: "rbracket", 0xDE: "quote",
}
VK.update({0x30 + d: str(d) for d in range(10)})
VK.update({0x41 + i: chr(ord("a") + i) for i in range(26)})
VK.update({0x60 + d: f"num{d}" for d in range(10)})
VK.update({0x70 + i: f"f{i + 1}" for i in range(24)})

# RAWMOUSE.usButtonFlags
_RAW_BUTTONS = {
    0x0001: ("mouse_left", True), 0x0002: ("mouse_left", False),
    0x0004: ("mouse_right", True), 0x0008: ("mouse_right", False),
    0x0010: ("mouse_middle", True), 0x0020: ("mouse_middle", False),
    0x0040: ("mouse_x1", True), 0x0080: ("mouse_x1", False),
    0x0100: ("mouse_x2", True), 0x0200: ("mouse_x2", False),
}
_RI_MOUSE_WHEEL = 0x0400
_WHEEL_DELTA = 120
_EVENT_BUTTONS = {"left": "mouse_left", "right": "mouse_right", "middle": "mouse_middle", "x1": "mouse_x1", "x2": "mouse_x2"}


def vk_name(vk: int) -> str:
    return VK.get(vk, f"vk{vk}")


def session_of(stem: str) -> str:
    return _SPLIT.sub("", stem)


def short_stem(stem: str, folder: str) -> str:
    """``recording_20250820_212408_Counter-Strike_2_ea30a554`` -> ``20250820_212408_ea30a554``."""
    stem = stem.removeprefix("recording_").replace(f"_{folder}_", "_")
    return re.sub(r"_+", "_", stem)


def _signed16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value >= 0x8000 else value


class D2ESource:
    name = "d2e"

    def discover(self, root: Path) -> list[Path]:
        return sorted(Path(root).rglob("*.mcap"))

    def convert(self, path: Path, out_root: Path, registry: WorldRegistry) -> RecordingMeta:
        path = Path(path)
        title = path.parent.name.replace("_", " ")
        world = registry.label_for(title)
        rec_id = f"d2e-{world}-{short_stem(path.stem, path.parent.name)}"
        root = Path(out_root) / rec_id
        root.mkdir(parents=True, exist_ok=True)

        msgs = [(m.topic, m.log_time, json.loads(m.data)) for m in McapReader(path).messages(TOPICS)]
        if not msgs:
            raise SystemExit(f"{path}: no messages on {sorted(TOPICS)}")
        origin = min(lt for _, lt, _ in msgs)

        def t_of(log_time: int) -> float:
            return round((log_time - origin) / 1e9, 6)

        notes = [f"source={path}"]
        events: list[InputEvent] = []
        has_raw = any(topic == "mouse/raw" for topic, _, _ in msgs)
        if not has_raw:
            notes.append("no mouse/raw topic: camera motion from absolute cursor positions")

        held: set[str] = set()
        repeats = absolute_raw = 0
        bins: dict[int, list[float]] = {}
        last_xy: tuple[int, int] | None = None
        titles: list[tuple[float, str]] = []
        for topic, lt, m in msgs:
            t = t_of(lt)
            if topic == "keyboard":
                code = vk_name(int(m["vk"]))
                if m["event_type"] == "press":
                    if code in held:
                        repeats += 1
                        continue
                    held.add(code)
                    events.append(InputEvent(t=t, kind=KEY, code=code, down=True))
                elif code in held:
                    held.discard(code)
                    events.append(InputEvent(t=t, kind=KEY, code=code, down=False))
            elif topic == "mouse/raw":
                if int(m.get("us_flags", 0)) & 0x1:
                    absolute_raw += 1
                elif m.get("last_x") or m.get("last_y"):
                    acc = bins.setdefault(int(t / MOVE_BIN_S + 1e-9), [0.0, 0.0])
                    acc[0] += m.get("last_x", 0)
                    acc[1] += m.get("last_y", 0)
                flags = int(m.get("button_flags", 0))
                for bit, (code, down) in _RAW_BUTTONS.items():
                    if flags & bit:
                        events.append(InputEvent(t=t, kind=BUTTON, code=code, down=down))
                if flags & _RI_MOUSE_WHEEL:
                    events.append(InputEvent(t=t, kind=WHEEL, dz=_signed16(int(m.get("button_data", 0))) / _WHEEL_DELTA))
            elif topic == "mouse" and not has_raw:
                x, y = int(m["x"]), int(m["y"])
                if m["event_type"] == "move":
                    if last_xy is not None:
                        events.append(InputEvent(t=t, kind=MOVE, dx=x - last_xy[0], dy=y - last_xy[1], x=x, y=y))
                    last_xy = (x, y)
                elif m["event_type"] == "click" and m.get("button") in _EVENT_BUTTONS:
                    events.append(InputEvent(t=t, kind=BUTTON, code=_EVENT_BUTTONS[m["button"]], down=bool(m.get("pressed"))))
                elif m["event_type"] == "scroll":
                    events.append(InputEvent(t=t, kind=WHEEL, dz=float(m.get("dy") or 0)))
            elif topic == "window":
                titles.append((t, str(m.get("title", ""))))
        events += [InputEvent(t=round(b * MOVE_BIN_S, 6), kind=MOVE, dx=dx, dy=dy) for b, (dx, dy) in bins.items()]
        if repeats:
            notes.append(f"auto-repeat presses dropped: {repeats}")
        if absolute_raw:
            notes.append(f"absolute raw mouse packets skipped: {absolute_raw}")

        # Focus: the longest-running title is the game.
        if titles:
            game = Counter(title for _, title in titles).most_common(1)[0][0]
            current = "game"
            for t, title in titles:
                code = "game" if title == game else (title[:80] or "untitled")
                if code != current:
                    events.append(InputEvent(t=t, kind=FOCUS, code=code))
                    current = code
            notes.append(f"window={game}")

        observations, videos, size = self._frames(msgs, t_of, path, root, notes)
        if not observations:
            raise SystemExit(f"{path}: no screen frames")
        span = observations[-1].t - observations[0].t
        fps = (len(observations) - 1) / span if span > 0 else 0.0
        gaps = sum(1 for a, b in zip(observations, observations[1:]) if b.t - a.t > 0.25)
        if gaps:
            notes.append(f"frame gaps over 250 ms: {gaps}")

        meta = RecordingMeta(
            recording_id=rec_id,
            world=world,
            source=self.name,
            stream=f"d2e:{world}",
            duration=round(observations[-1].t + (1.0 / fps if fps else 0.0), 6),
            frame_hz=round(fps, 3),
            width=size[0],
            height=size[1],
            mouse_scale=registry.get(world).mouse_scale(self.name),
            engine=f"{title} (human, desktop capture)",
            exact_state=False,
            session=f"d2e-{world}-{session_of(short_stem(path.stem, path.parent.name))}",
            notes=notes + [f"videos={','.join(videos) or 'none'}"],
        )
        write_recording(Recording(meta=meta, events=events, observations=observations), root)
        return meta

    def _frames(self, msgs, t_of, path: Path, root: Path, notes: list[str]):
        observations: list[Observation] = []
        names: dict[str, str] = {}
        size = (0, 0)
        embedded = 0
        frames_dir = root / "frames"
        for topic, lt, m in msgs:
            if topic != "screen":
                continue
            ref = m.get("media_ref") or {}
            uri = str(ref.get("uri", ""))
            if m.get("shape"):
                size = (int(m["shape"][0]), int(m["shape"][1]))
            t = t_of(lt)
            if uri.startswith("data:"):
                header, _, payload = uri.partition(",")
                ext = "png" if "png" in header else "jpg"
                frames_dir.mkdir(exist_ok=True)
                rel = f"frames/{len(observations):06d}.{ext}"
                (root / rel).write_bytes(base64.b64decode(payload))
                embedded += 1
                frame = rel
            else:
                if uri not in names:
                    names[uri] = self._link_video(path, uri, root, len(names))
                pts = ref.get("pts_ns")
                frame = f"{names[uri]}#t={(pts / 1e9 if pts is not None else t):.3f}"
            observations.append(Observation(t=t, index=len(observations), frame=frame))
        if embedded:
            notes.append(f"embedded frames written: {embedded}")
        observations.sort(key=lambda o: o.t)
        for i, obs in enumerate(observations):
            obs.index = i
        return observations, list(names.values()), size

    @staticmethod
    def _link_video(mcap: Path, uri: str, root: Path, k: int) -> str:
        target = Path(uri.removeprefix("file://"))
        if not target.is_absolute():
            target = mcap.parent / target
        ext = target.suffix or ".mkv"
        name = f"video{ext}" if k == 0 else f"video{k}{ext}"
        link = root / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if target.exists():
            os.symlink(target.resolve(), link)
        return name
