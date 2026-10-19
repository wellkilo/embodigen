"""Robot bodies.

Morphology is configuration, not code: a URDF-equivalent description plus an
action space. Everything else in the pipeline is morphology-agnostic.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULT_ROBOTS: dict[str, dict[str, Any]] = {
    "franka": {
        "name": "franka",
        "morphology": "arm",
        "dof": 7,
        "action_dim": 7,
        "reach": 0.85,
        "base": (0.0, 0.0, 0.0),
        "gripper_open": 0.04,
    },
    "mobile": {
        "name": "mobile",
        "morphology": "mobile",
        "dof": 3,
        "action_dim": 4,
        "reach": 1.20,
        "base": (0.0, 0.0, 0.0),
        "gripper_open": 0.04,
    },
    "humanoid": {
        "name": "humanoid",
        "morphology": "humanoid",
        "dof": 14,
        "action_dim": 7,
        "reach": 0.70,
        "base": (0.0, 0.0, 0.0),
        "gripper_open": 0.045,
    },
}


def load_robot(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    config_dir = config_dir or Path(__file__).resolve().parents[2] / "configs" / "robots"
    path = config_dir / f"{name}.toml"
    if path.exists():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        robot = dict(DEFAULT_ROBOTS.get(name, {}))
        robot.update(data)
        robot["base"] = tuple(robot.get("base", (0.0, 0.0, 0.0)))
        return robot
    if name in DEFAULT_ROBOTS:
        return dict(DEFAULT_ROBOTS[name])
    raise SystemExit(f"unknown robot {name!r}; known: {', '.join(sorted(DEFAULT_ROBOTS))}")


def available_robots() -> list[str]:
    return sorted(DEFAULT_ROBOTS)
