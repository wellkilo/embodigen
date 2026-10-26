"""ViZDoom backend: real Doom, driven through the same keyboard-mouse interface.

Needs the ``[env]`` extra (``pip install 'embodigen[env]'``). DOOM is one of
GameScaling's in-corpus worlds and its ViZDoom benchmark, which makes it the
natural first real engine: it exports exact depth, instance labels and game
variables on every tic, so rollouts carry engine truth like the robot
pipeline's ground-truth exporter does.

Doom advances in 35 Hz tics; recordings are timestamped in tics so the
canonicaliser resamples them like any other source.
"""

from __future__ import annotations

import math
import os
from typing import Any

from embodigen.gameplay.base import EnvFrame, InputState

try:  # pragma: no cover - exercised only with the [env] extra
    import vizdoom as vzd
except ImportError as exc:  # pragma: no cover
    raise ImportError("vizdoom is not installed") from exc

_KEY_BUTTONS = (
    ("w", "MOVE_FORWARD"),
    ("s", "MOVE_BACKWARD"),
    ("a", "MOVE_LEFT"),
    ("d", "MOVE_RIGHT"),
    ("mouse_left", "ATTACK"),
    ("e", "USE"),
    ("shift", "SPEED"),
)
_PICKUP_HINTS = ("medikit", "stimpack", "armor", "clip", "shell", "ammo", "health", "bonus", "cell", "rocket")


class VizDoomEnv:
    name = "vizdoom"
    tick_hz = 35.0
    deg_per_count = 0.12

    def __init__(
        self,
        width: int = 160,
        height: int = 120,
        scenario: str = "deathmatch.cfg",
        fov_deg: float = 90.0,
        episode_seconds: float = 0.0,
        **_: Any,
    ) -> None:
        res = getattr(vzd.ScreenResolution, f"RES_{width}X{height}", None)
        if res is None:
            options = sorted(n[4:] for n in dir(vzd.ScreenResolution) if n.startswith("RES_"))
            raise SystemExit(f"ViZDoom has no {width}x{height} mode; pick one of {', '.join(options)}")
        self.width, self.height = width, height
        self.fov = math.radians(fov_deg)
        game = vzd.DoomGame()
        path = scenario if os.path.isabs(scenario) else os.path.join(vzd.scenarios_path, scenario)
        game.load_config(path)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_screen_resolution(res)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_depth_buffer_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_available_buttons(
            [getattr(vzd.Button, b) for _, b in _KEY_BUTTONS] + [vzd.Button.TURN_LEFT_RIGHT_DELTA]
        )
        game.set_available_game_variables(
            [
                vzd.GameVariable.POSITION_X,
                vzd.GameVariable.POSITION_Y,
                vzd.GameVariable.ANGLE,
                vzd.GameVariable.HEALTH,
                vzd.GameVariable.KILLCOUNT,
                vzd.GameVariable.ITEMCOUNT,
            ]
        )
        if episode_seconds:
            game.set_episode_timeout(int(episode_seconds * self.tick_hz) + 1)
        game.set_mode(vzd.Mode.PLAYER)
        self.game = game
        self._initialised = False
        self._last: EnvFrame | None = None

    def reset(self, seed: int) -> None:
        if not self._initialised:
            self.game.set_seed(seed % (2**31))
            self.game.init()
            self._initialised = True
        else:
            self.game.set_seed(seed % (2**31))
        self.game.new_episode()

    def step(self, inp: InputState) -> None:
        keys = inp.keys if inp.focus == "game" else frozenset()
        dx = inp.dx if inp.focus == "game" else 0.0
        action = [1.0 if k in keys else 0.0 for k, _ in _KEY_BUTTONS]
        action.append(dx * self.deg_per_count)  # positive = turn right
        if not self.game.is_episode_finished():
            self.game.make_action(action, 1)

    def done(self) -> bool:
        return self.game.is_episode_finished() or self.game.is_player_dead()

    def observe(self, with_gt: bool = False) -> EnvFrame:
        st = self.game.get_state()
        if st is None:
            if self._last is None:
                raise RuntimeError("ViZDoom returned no state before the first frame")
            return self._last
        w, h = self.width, self.height
        px, py, angle, health, kills, items = (float(v) for v in st.game_variables)
        visible = []
        half = math.tan(self.fov / 2)
        for lab in st.labels:
            if lab.object_name == "DoomPlayer":
                continue
            name = lab.object_name.lower()
            kind = "pickup" if any(hint in name for hint in _PICKUP_HINTS) else "enemy"
            if lab.object_category and lab.object_category.lower() in ("item", "pickup"):
                kind = "pickup"
            cx = lab.x + lab.width / 2
            bearing = math.degrees(math.atan(((cx - w / 2) / (w / 2)) * half))
            dist = math.hypot(lab.object_position_x - px, lab.object_position_y - py) / 32.0
            visible.append(
                {
                    "id": int(lab.value),
                    "kind": kind,
                    "name": lab.object_name,
                    "bearing": round(bearing, 2),
                    "distance": round(dist, 3),
                }
            )
        state = {
            "pos": [round(px / 32.0, 4), round(py / 32.0, 4)],
            "yaw": round(angle, 3),
            "health": health,
            "score": int(kills + items),
            "tic": int(st.tic),
            "visible": sorted(visible, key=lambda e: e["distance"]),
        }
        frame = EnvFrame(
            rgb=st.screen_buffer.tobytes(),
            width=w,
            height=h,
            state=state,
            depth=[float(v) for v in st.depth_buffer.ravel()] if with_gt else None,
            labels=[int(v) for v in st.labels_buffer.ravel()] if with_gt else None,
        )
        self._last = frame
        return frame

    def close(self) -> None:
        if self._initialised:
            self.game.close()
            self._initialised = False
