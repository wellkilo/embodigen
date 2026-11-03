"""Optional interactive robot environments; importing this package needs only stdlib.

Install ``pip install -e '.[gym]'``, then ``env = make()``. The returned object
is a Gymnasium Env. MuJoCo and Gymnasium are loaded only by the backend.
"""

from __future__ import annotations

import importlib
import importlib.util

ENV_ID = "EmbodiGen/PickPlace-v0"


def available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("gymnasium", "mujoco", "numpy"))


def _backend():
    if not available():
        raise RuntimeError("交互 Gym 需要可选依赖，请在项目虚拟环境执行：pip install -e '.[gym]'")
    return importlib.import_module("embodigen.gym.mujoco_env")


def make(**kwargs):
    """Create the MuJoCo Cartesian parallel-gripper pick-and-place environment."""
    return _backend().PickPlaceEnv(**kwargs)


def register_envs() -> None:
    """Register ENV_ID for ``gymnasium.make`` (explicit, no import side effects)."""
    _backend().register_envs()
