"""Worlds are configuration, like robot bodies.

Each ``configs/worlds/<label>.toml`` names one canonical world: the label the
mixture builder counts shares by, the aliases and franchise the OOD splitter
de-duplicates against, and, for worlds we can run, the engine that drives
them. Source streams (one per uploader, channel or dataset shard) map onto
these labels; GameScaling maps 124 non-Minecraft streams onto 79 labels.

A file may instead hold a ``[[world]]`` array (a catalog), for datasets that
bring dozens of titles nobody will roll out locally.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CATEGORIES = ("AVG", "STG", "RPG", "RAC", "SIM", "OTH")
VIEWS = ("2D", "2.5D", "3D")

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "worlds"


def normalize_title(title: str) -> str:
    return re.sub(r"[^0-9a-z一-鿿]+", "", title.lower())


def slug(title: str) -> str:
    """Label for a title the registry does not know yet."""
    return re.sub(r"[^0-9a-z一-鿿]+", "-", title.lower()).strip("-") or "unknown"


# "Minecraft 1.21.8", "Battlefield 6 Open Beta" -> the title without the build
_VERSION = re.compile(r"[\s_-]+(v?\d+(\.\d+)+|open[\s_-]?beta|beta|demo|early[\s_-]?access)$", re.I)


@dataclass(frozen=True)
class WorldSpec:
    label: str
    title: str
    category: str = "OTH"
    view: str = "3D"
    aliases: tuple[str, ...] = ()
    franchise: str = ""
    emulates: str = ""  # "<platform>:<title>" for emulated games
    engine: str = ""  # env backend that can roll this world out, if any
    engine_args: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def names(self) -> set[str]:
        return {normalize_title(n) for n in (self.label, self.title, *self.aliases) if n}

    def mouse_scale(self, source: str) -> float:
        return float(self.sources.get(source, {}).get("mouse_scale", 1.0))

    def overlap(self, other: "WorldSpec") -> str:
        """Why two worlds count as the same game for OOD purposes ('' if they don't)."""
        if self.label == other.label:
            return "same label"
        if self.names & other.names:
            return "title/alias overlap"
        if self.franchise and self.franchise == other.franchise:
            return f"same franchise ({self.franchise})"
        if self.emulates and self.emulates == other.emulates:
            return f"same emulated title ({self.emulates})"
        return ""


def _spec(data: dict[str, Any], path: Path) -> WorldSpec:
    category = data.get("category", "OTH")
    view = data.get("view", "3D")
    if category not in CATEGORIES:
        raise SystemExit(f"{path}: category {category!r} not in {CATEGORIES}")
    if view not in VIEWS:
        raise SystemExit(f"{path}: view {view!r} not in {VIEWS}")
    return WorldSpec(
        label=data.get("label", path.stem),
        title=data.get("title", path.stem),
        category=category,
        view=view,
        aliases=tuple(data.get("aliases", ())),
        franchise=data.get("franchise", ""),
        emulates=data.get("emulates", ""),
        engine=data.get("engine", ""),
        engine_args=dict(data.get("engine_args", {})),
        sources={k: dict(v) for k, v in data.get("sources", {}).items()},
    )


class WorldRegistry:
    def __init__(self, worlds: dict[str, WorldSpec]) -> None:
        self.worlds = worlds
        self._by_name: dict[str, str] = {}
        for spec in worlds.values():
            for name in spec.names:
                self._by_name.setdefault(name, spec.label)

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "WorldRegistry":
        config_dir = Path(config_dir or CONFIG_DIR)
        worlds: dict[str, WorldSpec] = {}
        for path in sorted(config_dir.glob("*.toml")):
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            entries = data.get("world", [data])
            for entry in entries:
                if "world" in data and "label" not in entry:
                    raise SystemExit(f"{path}: every [[world]] entry needs a label")
                spec = _spec(entry, path)
                if spec.label in worlds:
                    raise SystemExit(f"{path}: duplicate world label {spec.label!r}")
                worlds[spec.label] = spec
        return cls(worlds)

    def __contains__(self, label: str) -> bool:
        return label in self.worlds

    def __getitem__(self, label: str) -> WorldSpec:
        try:
            return self.worlds[label]
        except KeyError:
            raise SystemExit(
                f"unknown world {label!r}; known: {', '.join(sorted(self.worlds))}. "
                f"Add configs/worlds/{label}.toml"
            ) from None

    def resolve(self, title: str) -> str | None:
        """Canonical label for a raw title or alias, if the registry knows it."""
        return self._by_name.get(normalize_title(title))

    def label_for(self, title: str) -> str:
        """Label a dataset title: known name, known name minus a version or
        beta suffix, else a slug (counted as its own unregistered world)."""
        title = title.replace("_", " ").strip()
        stripped = _VERSION.sub("", title)
        return self.resolve(title) or self.resolve(stripped) or slug(stripped)

    def get(self, label: str) -> WorldSpec:
        """Spec for a label, or a bare placeholder for worlds not yet registered."""
        return self.worlds.get(label) or WorldSpec(label=label, title=label)
