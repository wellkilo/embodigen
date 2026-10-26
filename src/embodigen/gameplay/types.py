"""The gameplay data contract.

Virtual worlds are the simulation environments here: a sample is what a
player saw and what they pressed, not what a robot joint did. Every stage of
the gameplay pipeline speaks these types, and like :mod:`embodigen.types` they
are plain dataclasses so a source adapter or a game engine can plug in without
importing anything heavy.

Time is in seconds from the start of a recording. Mouse motion is relative
(``dx`` right, ``dy`` down) in the source's own counts until the
canonicaliser rescales it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Event kinds. ``focus`` marks which application owns the foreground from that
# moment on; recordings without focus events are assumed to stay in the game.
KEY, BUTTON, MOVE, WHEEL, FOCUS = "key", "button", "move", "wheel", "focus"


@dataclass(frozen=True)
class InputEvent:
    t: float
    kind: str  # key | button | move | wheel | focus
    code: str = ""  # canonical key / button name, or the focused app
    down: bool = False
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    # Absolute cursor position when the source logs it; used to spot captures
    # where the game warps the cursor back to the centre every frame.
    x: float | None = None
    y: float | None = None

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": round(self.t, 6), "kind": self.kind}
        if self.code:
            row["code"] = self.code
        if self.kind in (KEY, BUTTON):
            row["down"] = self.down
        if self.kind == MOVE:
            row["dx"], row["dy"] = self.dx, self.dy
            if self.x is not None:
                row["x"], row["y"] = self.x, self.y
        if self.kind == WHEEL:
            row["dz"] = self.dz
        return row

    @staticmethod
    def from_json(row: dict[str, Any]) -> "InputEvent":
        return InputEvent(
            t=float(row["t"]),
            kind=row["kind"],
            code=row.get("code", ""),
            down=bool(row.get("down", False)),
            dx=float(row.get("dx", 0.0)),
            dy=float(row.get("dy", 0.0)),
            dz=float(row.get("dz", 0.0)),
            x=row.get("x"),
            y=row.get("y"),
        )


@dataclass
class Observation:
    """One captured frame. ``frame`` is relative to the recording directory,
    or ``<video>#t=<seconds>`` for sources that keep the original video."""

    t: float
    index: int
    frame: str
    depth: str | None = None  # engine truth, when the world can export it
    labels: str | None = None
    state: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": round(self.t, 6), "index": self.index, "frame": self.frame}
        for key in ("depth", "labels", "state"):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        return row

    @staticmethod
    def from_json(row: dict[str, Any]) -> "Observation":
        return Observation(
            t=float(row["t"]),
            index=int(row["index"]),
            frame=row["frame"],
            depth=row.get("depth"),
            labels=row.get("labels"),
            state=row.get("state"),
        )


@dataclass
class RecordingMeta:
    """Who played what, where it came from, and how to read it."""

    recording_id: str
    world: str  # canonical world label from the registry
    source: str  # vpt | rollout:<engine> | ...
    stream: str  # unit of the per-stream cap: one player, one channel, one seed family
    duration: float
    frame_hz: float
    width: int
    height: int
    app: str = "game"  # foreground name that counts as "in game"
    mouse_scale: float = 1.0  # source counts -> canonical counts
    engine: str = ""
    exact_state: bool = False  # True when state/depth/labels come from the engine
    split_hint: str = ""  # a source's own split ("val"/"test"), honoured by the splitter
    session: str = ""  # recordings cut from one play session; split together (default: own id)
    seed: int | None = None
    injected_lag_ms: float = 0.0  # audit trail for the desync demo; never read by validators
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(row: dict[str, Any]) -> "RecordingMeta":
        known = {f for f in RecordingMeta.__dataclass_fields__}
        return RecordingMeta(**{k: v for k, v in row.items() if k in known})


@dataclass
class Recording:
    meta: RecordingMeta
    events: list[InputEvent]
    observations: list[Observation]


@dataclass(frozen=True)
class ActionChunk:
    """One 200 ms action packet: a mouse displacement, a wheel step and the
    keys held in each of six 33 ms sub-steps."""

    dx: int
    dy: int
    dz: int
    keys: tuple[tuple[str, ...], ...]

    @property
    def idle(self) -> bool:
        return self.dx == 0 and self.dy == 0 and self.dz == 0 and not any(self.keys)


@dataclass
class Turn:
    """Observation at ``t`` followed by the action executed over the next chunk."""

    index: int
    t: float
    frame: str
    action: ActionChunk
    state: dict[str, Any] | None = None
    truncated_keys: int = 0


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]
    alignment: dict[str, Any] = field(default_factory=dict)
    realigned_ms: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Window:
    """A training clip: ``turns`` consecutive turns of one recording."""

    recording_id: str
    world: str
    stream: str
    start: int
    turns: int
    idle_frac: float

    @property
    def window_id(self) -> str:
        return f"{self.recording_id}:{self.start:06d}"
