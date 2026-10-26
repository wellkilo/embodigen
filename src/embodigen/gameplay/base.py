"""Gameplay stage protocols.

Same rule as :mod:`embodigen.modules.base`: a backend satisfies a protocol by
shape, so a game engine or a dataset adapter never has to import our classes.
Backends are registered in the shared ``BACKENDS`` table under the ``env``,
``source`` and ``annotate`` stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.types import RecordingMeta, Turn


@dataclass
class InputState:
    """What the player is doing during one control tick."""

    keys: frozenset[str] = frozenset()
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    focus: str = "game"


@dataclass
class EnvFrame:
    rgb: bytes
    width: int
    height: int
    state: dict[str, Any] = field(default_factory=dict)
    depth: list[float] | None = None  # planar, engine units
    labels: list[int] | None = None  # 0 background, 1 wall, >=2 entity


class WorldEnv(Protocol):
    """A virtual world we can play: one tick of input in, one frame out.

    ``state`` must at least carry ``visible``: a list of
    ``{"id", "kind", "bearing", "distance"}`` with bearing in degrees,
    positive to the right, so the scripted player can drive any world.
    """

    name: str
    tick_hz: float
    deg_per_count: float  # view rotation per mouse count

    def reset(self, seed: int) -> None: ...

    def step(self, inp: InputState) -> None: ...

    def observe(self, with_gt: bool) -> EnvFrame: ...

    def done(self) -> bool: ...

    def close(self) -> None: ...


class SourceAdapter(Protocol):
    """Converts one dataset's native recordings into the raw layout."""

    name: str

    def discover(self, root: Path) -> list[Path]: ...

    def convert(self, path: Path, out_root: Path, registry: WorldRegistry) -> RecordingMeta | list[RecordingMeta]:
        """One native file -> one recording, or several when the file mixes
        human play with something else (the parts share a ``session``)."""
        ...


class Annotator(Protocol):
    """Writes a short retrospective rationale for an executed action.

    It explains what the player did; it never changes the action label.
    """

    name: str
    prompt_version: str

    def rationale(self, turn: Turn, context: list[Turn], world: str) -> str: ...
