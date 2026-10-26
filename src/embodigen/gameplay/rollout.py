"""Engine rollouts: play a world and record it like a human capture.

The scripted player emits raw input events (key down/up, mouse counts,
focus changes) at the engine's tick rate, and the driver writes the same
``raw/<recording_id>/`` layout a human recording converts into. Downstream
stages cannot tell the two apart except through ``exact_state``.

Profiles exist to exercise the quality gate on purpose:

* ``player``     -- plays the objective, with short pauses;
* ``afk``        -- mostly idle, should be rejected for idle windows;
* ``distracted`` -- alt-tabs to another app and types there, should be
  rejected for non-game foreground input.
"""

from __future__ import annotations

import math
import random
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodigen.gameplay.base import InputState
from embodigen.gameplay.recording import write_recording
from embodigen.gameplay.types import (
    BUTTON,
    FOCUS,
    KEY,
    MOVE,
    InputEvent,
    Observation,
    Recording,
    RecordingMeta,
)
from embodigen.pngio import write_png

PROFILES = {
    # idle_rate: idle bouts started per second; idle_len: bout length range (s)
    "player": {"idle_rate": 0.04, "idle_len": (0.8, 2.5), "alt_tab_rate": 0.0},
    "afk": {"idle_rate": 0.6, "idle_len": (6.0, 14.0), "alt_tab_rate": 0.0},
    "distracted": {"idle_rate": 0.04, "idle_len": (0.8, 2.5), "alt_tab_rate": 0.04},
}


class ScriptedPlayer:
    """A noisy objective-seeking player that only sees ``state['visible']``."""

    def __init__(self, rng: random.Random, deg_per_count: float, tick_hz: float, profile: str = "player") -> None:
        if profile not in PROFILES:
            raise SystemExit(f"unknown profile {profile!r}; known: {', '.join(PROFILES)}")
        self.rng = rng
        self.deg_per_count = deg_per_count
        self.dt = 1.0 / tick_hz
        self.cfg = PROFILES[profile]
        self.idle_until = 0.0
        self.away_until = 0.0
        self.fire_ticks = 0
        self.strafe: str | None = None
        self.strafe_until = 0.0
        self.wander_phase = rng.uniform(0, 2 * math.pi)
        self.wander_dir = rng.choice((-1, 1))
        self.noise = 0.0
        self.last_pos: list[float] | None = None
        self.stuck = 0.0
        self.extra_key: str | None = None
        self.extra_until = 0.0

    def act(self, state: dict[str, Any], t: float) -> InputState:
        rng, dt = self.rng, self.dt
        if t < self.away_until:
            # Typing in another app: input happens, but not in the game.
            keys = frozenset({rng.choice("abcdefghijklmnopqrstuvwxyz")}) if rng.random() < 0.4 else frozenset()
            return InputState(keys=keys, dx=rng.gauss(0, 3), dy=rng.gauss(0, 3), focus="browser")
        if t < self.idle_until:
            return InputState()
        if rng.random() < self.cfg["alt_tab_rate"] * dt:
            self.away_until = t + rng.uniform(2.5, 6.0)
            return InputState(focus="browser")
        if rng.random() < self.cfg["idle_rate"] * dt:
            self.idle_until = t + rng.uniform(*self.cfg["idle_len"])
            return InputState()

        visible = state.get("visible", [])
        objective = state.get("objective", "hunt")
        prefer = {"hunt": "enemy", "gather": "pickup", "exit": "exit"}.get(objective, "enemy")
        targets = [v for v in visible if v["kind"] == prefer] or [v for v in visible if v["kind"] == "enemy"] \
            or [v for v in visible if v["kind"] in ("pickup", "exit")]
        target = min(targets, key=lambda v: v["distance"]) if targets else None

        # Smoothed hand tremor so the mouse never moves in perfectly clean steps.
        self.noise = 0.85 * self.noise + rng.gauss(0, 0.6)
        keys: set[str] = set()
        if target is not None:
            turn_deg = 0.22 * target["bearing"] + self.noise
            aligned = abs(target["bearing"]) < 5.0
            if abs(target["bearing"]) < 30 and target["distance"] > 1.1:
                keys.add("w")
                if target["distance"] > 4 and rng.random() < 0.6:
                    keys.add("shift")
            if target["kind"] == "enemy" and aligned and self.fire_ticks == 0 and rng.random() < 0.5:
                self.fire_ticks = 3
        else:
            self.wander_phase += dt * 0.9
            turn_deg = self.wander_dir * (1.2 + 1.5 * math.sin(self.wander_phase)) + self.noise
            keys.add("w")
            if rng.random() < 0.2 * dt:
                self.wander_dir *= -1
        if self.fire_ticks > 0:
            keys.add("mouse_left")
            self.fire_ticks -= 1

        # Getting stuck on a wall: back off and swing round, like people do.
        pos = state.get("pos")
        if pos is not None and self.last_pos is not None and "w" in keys:
            moved = math.hypot(pos[0] - self.last_pos[0], pos[1] - self.last_pos[1])
            self.stuck = self.stuck + dt if moved < 0.01 else 0.0
        self.last_pos = pos
        if self.stuck > 0.4:
            keys.discard("w")
            keys.add("s")
            turn_deg += 9.0 * self.wander_dir

        if t >= self.strafe_until:
            self.strafe = rng.choice(("a", "d")) if rng.random() < 0.25 else None
            self.strafe_until = t + rng.uniform(0.3, 1.2)
        if self.strafe:
            keys.add(self.strafe)
        if t >= self.extra_until:
            self.extra_key = None
            if rng.random() < 0.6 * dt:
                self.extra_key = rng.choice(("space", "e", "1", "2", "r", "ctrl"))
                self.extra_until = t + rng.uniform(0.05, 0.25)
        if self.extra_key:
            keys.add(self.extra_key)

        dx = turn_deg / self.deg_per_count
        dy = rng.gauss(0, 0.8) - 0.15 * state.get("pitch", 0.0) / self.deg_per_count
        return InputState(keys=frozenset(keys), dx=dx, dy=dy)


class EventRecorder:
    """Turns per-tick input states into a raw event log."""

    def __init__(self, lag_s: float = 0.0, drift: float = 0.0) -> None:
        self.lag = lag_s
        self.drift = drift  # extra seconds of lag per second of play (clock drift)
        self.held: frozenset[str] = frozenset()
        self.focus = "game"
        self.rx = self.ry = 0.0
        self.events: list[InputEvent] = []

    def record(self, t: float, inp: InputState) -> None:
        t = round(t + self.lag + self.drift * t, 6)
        if inp.focus != self.focus:
            self.events.append(InputEvent(t=t, kind=FOCUS, code=inp.focus))
            self.focus = inp.focus
        for code in sorted(self.held - inp.keys):
            self.events.append(InputEvent(t=t, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=False))
        for code in sorted(inp.keys - self.held):
            self.events.append(InputEvent(t=t, kind=BUTTON if code.startswith("mouse_") else KEY, code=code, down=True))
        self.held = inp.keys
        # Mice report whole counts; keep the remainder for the next report.
        self.rx += inp.dx
        self.ry += inp.dy
        cx, cy = int(self.rx), int(self.ry)
        if cx or cy:
            self.rx -= cx
            self.ry -= cy
            self.events.append(InputEvent(t=t, kind=MOVE, dx=float(cx), dy=float(cy)))


@dataclass
class RolloutSpec:
    world: str
    engine: str
    engine_args: dict[str, Any]
    seconds: float
    seed: int
    profile: str = "player"
    lag_ms: float = 0.0
    drift_ms_per_min: float = 0.0
    width: int = 160
    height: int = 100
    gt: bool = True
    stream: str = ""


def rollout(env: Any, spec: RolloutSpec, out_root: Path, recording_id: str) -> RecordingMeta:
    """Play ``spec.seconds`` of a world and write it under ``out_root/recording_id``."""
    root = Path(out_root) / recording_id
    if root.is_dir():
        # Re-recording under the same id must not leave frames of a longer take behind.
        shutil.rmtree(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    if spec.gt:
        (root / "depth").mkdir(exist_ok=True)
        (root / "labels").mkdir(exist_ok=True)

    rng = random.Random(spec.seed * 7919 + 17)
    env.reset(spec.seed)
    player = ScriptedPlayer(rng, env.deg_per_count, env.tick_hz, spec.profile)
    recorder = EventRecorder(lag_s=spec.lag_ms / 1000.0, drift=spec.drift_ms_per_min / 60000.0)
    obs: list[Observation] = []
    # Engine truth is exported on the 5 Hz grid the training turns sit on.
    gt_stride = max(1, round(env.tick_hz / 5.0))

    ticks = int(spec.seconds * env.tick_hz)
    k = 0
    for k in range(ticks):
        t = k / env.tick_hz
        on_grid = k % gt_stride == 0
        frame = env.observe(with_gt=spec.gt and on_grid)
        name = f"{k:06d}"
        write_png(root / "frames" / f"{name}.png", frame.width, frame.height, frame.rgb)
        row = Observation(t=t, index=k, frame=f"frames/{name}.png")
        if on_grid:
            row.state = frame.state
            if frame.depth is not None:
                (root / "depth" / f"{name}.f32").write_bytes(struct.pack(f"<{len(frame.depth)}f", *frame.depth))
                row.depth = f"depth/{name}.f32"
            if frame.labels is not None:
                (root / "labels" / f"{name}.u16").write_bytes(struct.pack(f"<{len(frame.labels)}H", *frame.labels))
                row.labels = f"labels/{name}.u16"
        obs.append(row)
        inp = player.act(frame.state, t)
        recorder.record(t, inp)
        env.step(inp)
        if env.done():
            k += 1
            break
    else:
        k = ticks

    duration = k / env.tick_hz
    meta = RecordingMeta(
        recording_id=recording_id,
        world=spec.world,
        source=f"rollout:{spec.engine}",
        stream=spec.stream or f"{spec.engine}:{spec.world}:{spec.profile}",
        duration=round(duration, 6),
        frame_hz=env.tick_hz,
        width=spec.width,
        height=spec.height,
        engine=env.name,
        exact_state=True,
        seed=spec.seed,
        injected_lag_ms=spec.lag_ms,
        notes=[f"profile={spec.profile}"]
        + ([f"injected_drift_ms_per_min={spec.drift_ms_per_min:g}"] if spec.drift_ms_per_min else []),
    )
    events = [e for e in recorder.events if 0.0 <= e.t < duration]
    write_recording(Recording(meta=meta, events=events, observations=obs), root)
    return meta
