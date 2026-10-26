"""Open-P2P (Pixels2Play) gameplay -> raw recordings.

Open-P2P (https://huggingface.co/datasets/elefantai/p2p-full-data) unpacks to
one directory per recording::

    <uuid>/video.mp4         raw capture
    <uuid>/192x192.mp4       the resized copy P2P trains on
    <uuid>/annotation.proto  serialized ``VideoAnnotation`` protobuf

with one ``FrameAnnotation`` per video frame (20 fps). The fields read here,
by number, from ``video_annotation.proto`` in github.com/elefant-ai/open-p2p::

    VideoAnnotation   2 metadata, 3 frame_annotations[]
    Metadata          1 id, 3 user, 5 frames_per_second (float), 6 env {1 env, 2 env_subtype},
                      10 capture_device_specs {3 window_specs {1 width, 2 height}}
    FrameAnnotation   1 user_action, 6 system_action, 7 frame_text_annotation[] {1 instruction}
    LowLevelAction    2 keyboard {1 keys[]}, 3 is_known, 4 mouse, 1 mouse_deprecated
    MouseAction       3 mouse_delta_px {1 x, 2 y}, 4 scroll_delta_px {1 x, 2 y}, 9 buttons_down[] ("0", "1", "2")

Only frames whose ``user_action`` is known are human play. Frames driven by
P2P's inference server (``system_action``, their correction data) or with no
known action split a file into separate recordings, one per contiguous human
run, sharing a ``session`` so they stay on one side of every split. Runs
shorter than one training window are dropped and counted in the notes.
"""

from __future__ import annotations

import os
from pathlib import Path

from embodigen.gameplay.actions import canonical_key
from embodigen.gameplay.recording import write_recording
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.sources import protowire as pw
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

P2P_FPS = 20.0
MIN_RUN_S = 4.0  # one 20-turn window
_BUTTONS = {"0": "mouse_left", "1": "mouse_right", "2": "mouse_middle"}
_VIDEOS = ("video.mp4", "192x192.mp4")


def _vec(raw: bytes | None, cast=pw.int32) -> tuple[float, float]:
    if not raw:
        return 0.0, 0.0
    f = pw.parse(raw)
    return cast(pw.first(f, 1, 0)), cast(pw.first(f, 2, 0))


def _ints(values: list) -> list[int]:
    """A repeated int32 field, packed (bytes) or not."""
    out: list[int] = []
    for v in values:
        if isinstance(v, bytes):
            off = 0
            while off < len(v):
                n, off = pw._varint(v, off)
                out.append(pw.int32(n))
        else:
            out.append(pw.int32(v))
    return out


def frame_action(raw: bytes) -> tuple[str, set[str], float, float, float, str]:
    """(who, held keys+buttons, dx, dy, wheel, instruction) for one FrameAnnotation."""
    fa = pw.parse(raw)
    user = pw.parse(pw.first(fa, 1, b""))
    system = pw.parse(pw.first(fa, 6, b""))
    instruction = ""
    for text in fa.get(7, []):
        instruction = pw.string(pw.parse(text), 1) or instruction
    if pw.first(system, 3, 0):
        return "system", set(), 0.0, 0.0, 0.0, instruction
    if not pw.first(user, 3, 0):
        return "unknown", set(), 0.0, 0.0, 0.0, instruction
    held = {canonical_key(k.decode("utf-8", "replace")) for k in pw.parse(pw.first(user, 2, b"")).get(1, [])}
    dx = dy = wheel = 0.0
    if 4 in user:
        mouse = pw.parse(pw.first(user, 4))
        dx, dy = _vec(pw.first(mouse, 3))
        wheel = _vec(pw.first(mouse, 4))[1]
        held |= {_BUTTONS[b.decode()] for b in mouse.get(9, []) if b.decode() in _BUTTONS}
    elif 1 in user:  # DeprecatedMouseAction: 3 dx, 4 dy (double), 5 buttons[]
        mouse = pw.parse(pw.first(user, 1))
        dx = pw.double(pw.first(mouse, 3)) if 3 in mouse else 0.0
        dy = pw.double(pw.first(mouse, 4)) if 4 in mouse else 0.0
        held |= {_BUTTONS[str(b)] for b in _ints(mouse.get(5, [])) if str(b) in _BUTTONS}
    return "user", held, dx, dy, wheel, instruction


def human_runs(who: list[str], min_frames: int) -> tuple[list[tuple[int, int]], int]:
    """Contiguous [start, end) runs of human frames at least ``min_frames`` long, and the short ones dropped."""
    runs, dropped, start = [], 0, None
    for i, w in enumerate(who + ["end"]):
        if w == "user" and start is None:
            start = i
        elif w != "user" and start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            else:
                dropped += 1
            start = None
    return runs, dropped


class P2PSource:
    name = "p2p"

    def __init__(self, min_run_s: float = MIN_RUN_S) -> None:
        self.min_run_s = min_run_s

    def discover(self, root: Path) -> list[Path]:
        return sorted(Path(root).rglob("annotation.proto"))

    def convert(self, path: Path, out_root: Path, registry: WorldRegistry) -> list[RecordingMeta]:
        path = Path(path)
        top = pw.parse(path.read_bytes())
        meta_f = pw.parse(pw.first(top, 2, b""))
        uuid = pw.string(meta_f, 1) or path.parent.name
        user = pw.string(meta_f, 3)
        fps = pw.float32(pw.first(meta_f, 5)) if 5 in meta_f else P2P_FPS
        fps = fps if fps > 0 else P2P_FPS
        env_f = pw.parse(pw.first(meta_f, 6, b""))
        env, subtype = pw.string(env_f, 1), pw.string(env_f, 2)
        title = f"{env} {subtype}" if env.lower() == "roblox" and subtype else env
        world = registry.label_for(title or "unknown")

        video = next((name for name in _VIDEOS if (path.parent / name).exists()), "")
        width = height = 0
        if video == "video.mp4":
            specs = pw.parse(pw.first(meta_f, 10, b""))
            window = pw.parse(pw.first(specs, 3, b""))
            width, height = pw.first(window, 1, 0), pw.first(window, 2, 0)
        elif video:
            width = height = 192

        frames = [frame_action(raw) for raw in top.get(3, [])]
        who = [f[0] for f in frames]
        runs, short = human_runs(who, int(round(self.min_run_s * fps)))
        base_notes = [f"source={path}", f"env={env}" + (f"/{subtype}" if subtype else "")]
        counts = {k: who.count(k) for k in ("user", "system", "unknown") if who.count(k)}
        base_notes.append("frames " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if short:
            base_notes.append(f"human runs shorter than {self.min_run_s:g}s dropped: {short}")
        if not video:
            base_notes.append("video missing: frames are references only")

        metas = []
        for k, (start, end) in enumerate(runs):
            rec_id = f"p2p-{uuid}" if len(runs) == 1 else f"p2p-{uuid}-{k:02d}"
            root = Path(out_root) / rec_id
            root.mkdir(parents=True, exist_ok=True)
            if video:
                link = root / "video.mp4"
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink((path.parent / video).resolve(), link)
            events, observations = self._run(frames, start, end, fps)
            meta = RecordingMeta(
                recording_id=rec_id,
                world=world,
                source=self.name,
                stream=f"p2p:{user or uuid}",
                duration=round((end - start) / fps, 6),
                frame_hz=fps,
                width=int(width),
                height=int(height),
                mouse_scale=registry.get(world).mouse_scale(self.name),
                engine=f"{title or 'unknown'} (human)",
                exact_state=False,
                session=f"p2p-{uuid}",
                notes=base_notes + [f"frames {start}..{end - 1} of {len(frames)}", f"video={video or 'none'}"],
            )
            write_recording(Recording(meta=meta, events=events, observations=observations), root)
            metas.append(meta)
        return metas

    @staticmethod
    def _run(frames, start: int, end: int, fps: float) -> tuple[list[InputEvent], list[Observation]]:
        events: list[InputEvent] = []
        observations: list[Observation] = []
        held: set[str] = set()
        for i in range(start, end):
            _, now, dx, dy, wheel, instruction = frames[i]
            t = round((i - start) / fps, 6)
            for code in sorted(held - now):
                events.append(InputEvent(t=t, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=False))
            for code in sorted(now - held):
                events.append(InputEvent(t=t, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=True))
            held = now
            if dx or dy:
                events.append(InputEvent(t=t, kind=MOVE, dx=dx, dy=dy))
            if wheel:
                events.append(InputEvent(t=t, kind=WHEEL, dz=1.0 if wheel > 0 else -1.0))
            state = {"instruction": instruction} if instruction else {}
            observations.append(Observation(t=t, index=i - start, frame=f"video.mp4#t={i / fps:.3f}", state=state))
        t_end = round((end - start) / fps, 6)
        for code in sorted(held):
            events.append(InputEvent(t=t_end, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=False))
        return events, observations
