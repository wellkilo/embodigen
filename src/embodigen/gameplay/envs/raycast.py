"""Reference virtual world: a small first-person raycaster, standard library only.

It plays the role the mock renderer plays in the robot pipeline: it proves
the data path end to end -- keyboard-mouse in, frames plus exact depth,
instance labels and state out -- without a GPU or a game install. Each world
config picks a layout, a palette and an objective, so a handful of configs
give the mixture builder a genuinely multi-world corpus to work with.

Controls: ``w/a/s/d`` move, ``shift`` runs, mouse ``dx`` turns, ``dy`` looks
up/down, ``mouse_left`` fires. Units are cells (~1 m).
"""

from __future__ import annotations

import math
import random
from typing import Any

from embodigen.gameplay.base import EnvFrame, InputState

THEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "stone": {"wall": (150, 138, 120), "floor": (92, 84, 74), "ceil": (58, 62, 74), "mortar": (60, 55, 50)},
    "tech": {"wall": (96, 132, 168), "floor": (54, 62, 70), "ceil": (26, 30, 40), "mortar": (210, 220, 230)},
    "forest": {"wall": (86, 128, 70), "floor": (104, 86, 56), "ceil": (150, 186, 214), "mortar": (52, 76, 40)},
    "lava": {"wall": (120, 60, 50), "floor": (70, 36, 30), "ceil": (36, 20, 20), "mortar": (230, 140, 40)},
}
ENTITY_STYLE = {
    "enemy": {"size": 0.8, "aspect": 0.55, "color": (200, 40, 40), "accent": (250, 220, 60)},
    "pickup": {"size": 0.35, "aspect": 1.0, "color": (240, 200, 40), "accent": (255, 255, 200)},
    "exit": {"size": 1.0, "aspect": 0.6, "color": (40, 200, 90), "accent": (200, 255, 210)},
}


def _layout(kind: str, size: int, rng: random.Random) -> list[list[int]]:
    """Grid of wall ids: 0 is open floor, >0 a wall (the id varies its tint)."""
    grid = [[0] * size for _ in range(size)]
    for i in range(size):
        grid[0][i] = grid[size - 1][i] = grid[i][0] = grid[i][size - 1] = 1
    if kind == "maze":
        grid = [[1] * size for _ in range(size)]
        stack = [(1, 1)]
        grid[1][1] = 0
        while stack:
            x, y = stack[-1]
            options = [
                (x + dx, y + dy, dx, dy)
                for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 0 < x + dx < size - 1 and 0 < y + dy < size - 1 and grid[y + dy][x + dx]
            ]
            if not options:
                stack.pop()
                continue
            nx, ny, dx, dy = rng.choice(options)
            grid[y + dy // 2][x + dx // 2] = 0
            grid[ny][nx] = 0
            stack.append((nx, ny))
        # Knock out some walls so the maze has loops and sight lines.
        for _ in range(size * 2):
            x, y = rng.randrange(1, size - 1), rng.randrange(1, size - 1)
            grid[y][x] = 0
    elif kind == "rooms":
        step = max(4, size // 3)
        for k in range(step, size - 1, step):
            for i in range(size):
                grid[k][i] = grid[i][k] = 1
        for k in range(step, size - 1, step):
            for lo in range(0, size - 1, step):
                door = min(size - 2, lo + 1 + rng.randrange(max(1, step - 1)))
                grid[k][door] = grid[door][k] = 0
    else:  # arena: open floor with pillars
        for _ in range(int(size * size * 0.06)):
            x, y = rng.randrange(2, size - 2), rng.randrange(2, size - 2)
            grid[y][x] = 1
    for y in range(size):
        for x in range(size):
            if grid[y][x]:
                grid[y][x] = 1 + rng.randrange(4)
    return grid


class RaycastWorld:
    name = "mock-raycast-v0"
    tick_hz = 30.0
    deg_per_count = 0.12

    def __init__(
        self,
        width: int = 160,
        height: int = 100,
        layout: str = "arena",
        theme: str = "stone",
        objective: str = "hunt",
        size: int = 15,
        enemies: int = 5,
        pickups: int = 4,
        fov_deg: float = 66.0,
        enemy_speed: float = 0.6,
        **_: Any,
    ) -> None:
        if theme not in THEMES:
            raise SystemExit(f"unknown raycast theme {theme!r}; known: {', '.join(THEMES)}")
        self.width, self.height = width, height
        self.layout, self.theme, self.objective = layout, THEMES[theme], objective
        self.size = size if layout != "maze" else size | 1
        self.n_enemies, self.n_pickups = enemies, pickups
        self.plane = math.tan(math.radians(fov_deg) / 2)
        self.fov_deg = fov_deg
        self.enemy_speed = enemy_speed
        self.reset(0)

    # -- world ---------------------------------------------------------------

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.grid = _layout(self.layout, self.size, self.rng)
        self.x, self.y = self._free_cell()
        self.yaw = self.rng.uniform(-math.pi, math.pi)
        self.pitch = 0.0
        self.health, self.score, self.deaths, self.time = 100.0, 0, 0, 0.0
        self.cooldown = 0.0
        self.entities: list[dict[str, Any]] = []
        kinds = ["enemy"] * self.n_enemies + ["pickup"] * self.n_pickups
        if self.objective == "exit":
            kinds.append("exit")
        for i, kind in enumerate(kinds):
            ex, ey = self._free_cell(away_from=(self.x, self.y))
            self.entities.append({"id": i + 2, "kind": kind, "x": ex, "y": ey, "alive": True, "respawn": 0.0})

    def _wall(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.size and 0 <= iy < self.size):
            return True
        return self.grid[iy][ix] > 0

    def _free_cell(self, away_from: tuple[float, float] | None = None) -> tuple[float, float]:
        for _ in range(500):
            ix, iy = self.rng.randrange(1, self.size - 1), self.rng.randrange(1, self.size - 1)
            if self.grid[iy][ix]:
                continue
            cx, cy = ix + 0.5, iy + 0.5
            if away_from and math.hypot(cx - away_from[0], cy - away_from[1]) < 3.0:
                continue
            return cx, cy
        return 1.5, 1.5

    def _dir(self) -> tuple[float, float, float, float]:
        dx, dy = math.cos(self.yaw), math.sin(self.yaw)
        return dx, dy, dy, -dx  # forward, right

    def step(self, inp: InputState) -> None:
        dt = 1.0 / self.tick_hz
        self.time += dt
        self.cooldown = max(0.0, self.cooldown - dt)
        in_game = inp.focus == "game"
        keys = inp.keys if in_game else frozenset()
        if in_game:
            self.yaw -= math.radians(inp.dx * self.deg_per_count)
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.pitch = max(-0.6, min(0.6, self.pitch + math.radians(inp.dy * self.deg_per_count)))

        fx, fy, rx, ry = self._dir()
        fwd = ("w" in keys) - ("s" in keys)
        strafe = ("d" in keys) - ("a" in keys)
        speed = 2.5 * (1.6 if "shift" in keys else 1.0) * dt
        mx = (fwd * fx + strafe * rx) * speed
        my = (fwd * fy + strafe * ry) * speed
        r = 0.2
        if not self._wall(self.x + mx + math.copysign(r, mx or 1), self.y):
            self.x += mx
        if not self._wall(self.x, self.y + my + math.copysign(r, my or 1)):
            self.y += my

        if "mouse_left" in keys and self.cooldown == 0.0:
            self.cooldown = 0.35
            target = self._aimed_enemy()
            if target is not None:
                target["alive"], target["respawn"] = False, 2.0
                self.score += 1

        for ent in self.entities:
            if not ent["alive"]:
                ent["respawn"] -= dt
                if ent["respawn"] <= 0:
                    ent["x"], ent["y"] = self._free_cell(away_from=(self.x, self.y))
                    ent["alive"] = True
                continue
            dist = math.hypot(ent["x"] - self.x, ent["y"] - self.y)
            if ent["kind"] == "enemy":
                if 0.8 < dist < 6.0:
                    step = self.enemy_speed * dt / dist
                    nx, ny = ent["x"] + (self.x - ent["x"]) * step, ent["y"] + (self.y - ent["y"]) * step
                    if not self._wall(nx, ny):
                        ent["x"], ent["y"] = nx, ny
                if dist < 1.0:
                    self.health -= 10.0 * dt
            elif dist < 0.5:
                ent["alive"], ent["respawn"] = False, 3.0
                self.score += 1
        if self.health <= 0:
            self.deaths += 1
            self.health = 100.0
            self.x, self.y = self._free_cell()

    def _aimed_enemy(self) -> dict[str, Any] | None:
        wall = self._cast(math.cos(self.yaw), math.sin(self.yaw))[0]
        best = None
        for ent in self.entities:
            if ent["kind"] != "enemy" or not ent["alive"]:
                continue
            depth, lateral = self._to_camera(ent)
            if depth <= 0.1 or depth > wall:
                continue
            half = ENTITY_STYLE["enemy"]["size"] * ENTITY_STYLE["enemy"]["aspect"] / 2
            if abs(lateral) < half and (best is None or depth < best[0]):
                best = (depth, ent)
        return best[1] if best else None

    def done(self) -> bool:
        return False

    def close(self) -> None:
        pass

    # -- rendering -----------------------------------------------------------

    def _to_camera(self, ent: dict[str, Any]) -> tuple[float, float]:
        fx, fy, rx, ry = self._dir()
        ox, oy = ent["x"] - self.x, ent["y"] - self.y
        return ox * fx + oy * fy, ox * rx + oy * ry

    def _cast(self, ray_x: float, ray_y: float) -> tuple[float, int, float, int]:
        """DDA: (perpendicular distance, side, texture u, wall id)."""
        map_x, map_y = int(self.x), int(self.y)
        delta_x = abs(1.0 / ray_x) if ray_x else 1e30
        delta_y = abs(1.0 / ray_y) if ray_y else 1e30
        if ray_x < 0:
            step_x, side_x = -1, (self.x - map_x) * delta_x
        else:
            step_x, side_x = 1, (map_x + 1.0 - self.x) * delta_x
        if ray_y < 0:
            step_y, side_y = -1, (self.y - map_y) * delta_y
        else:
            step_y, side_y = 1, (map_y + 1.0 - self.y) * delta_y
        side = 0
        for _ in range(self.size * 3):
            if side_x < side_y:
                side_x += delta_x
                map_x += step_x
                side = 0
            else:
                side_y += delta_y
                map_y += step_y
                side = 1
            if not (0 <= map_x < self.size and 0 <= map_y < self.size) or self.grid[map_y][map_x]:
                break
        dist = (side_x - delta_x) if side == 0 else (side_y - delta_y)
        dist = max(dist, 1e-3)
        hit = self.y + dist * ray_y if side == 0 else self.x + dist * ray_x
        wall_id = self.grid[map_y][map_x] if 0 <= map_x < self.size and 0 <= map_y < self.size else 1
        return dist, side, hit - math.floor(hit), wall_id

    def observe(self, with_gt: bool = False) -> EnvFrame:
        w, h = self.width, self.height
        fx, fy, rx, ry = self._dir()
        horizon = h / 2 - self.pitch * h / 2
        rgb = bytearray(w * h * 3)
        depth = [0.0] * (w * h) if with_gt else None
        labels = [0] * (w * h) if with_gt else None
        theme = self.theme
        zbuf = [0.0] * w

        ceil_rows: list[bytes] = []
        for r in range(h):
            k = max(0.35, 1.0 - r / (h * 1.6))
            ceil_rows.append(bytes(int(c * k) for c in theme["ceil"]))
        floor_a = theme["floor"]
        floor_b = tuple(int(c * 0.78) for c in floor_a)

        for col in range(w):
            cam = 2.0 * col / w - 1.0
            ray_x, ray_y = fx + rx * self.plane * cam, fy + ry * self.plane * cam
            dist, side, u, wall_id = self._cast(ray_x, ray_y)
            zbuf[col] = dist
            line = h / dist
            top, bottom = horizon - line / 2, horizon + line / 2
            shade = max(0.25, 1.0 - dist / 12.0) * (0.78 if side else 1.0)
            tint = 0.82 + 0.06 * wall_id
            base = bytes(min(255, int(c * shade * tint)) for c in theme["wall"])
            mortar = bytes(int(c * shade) for c in theme["mortar"])
            for r in range(h):
                o = (r * w + col) * 3
                if r < top:
                    rgb[o : o + 3] = ceil_rows[r]
                    if depth is not None:
                        depth[r * w + col] = h / (2.0 * max(horizon - r, 0.5))
                    continue
                if r >= bottom:
                    rd = h / (2.0 * max(r - horizon, 0.5))
                    wx, wy = self.x + rd * ray_x, self.y + rd * ray_y
                    tone = floor_a if (int(math.floor(wx)) + int(math.floor(wy))) & 1 else floor_b
                    fk = max(0.3, 1.0 - rd / 12.0)
                    rgb[o] = int(tone[0] * fk)
                    rgb[o + 1] = int(tone[1] * fk)
                    rgb[o + 2] = int(tone[2] * fk)
                    if depth is not None:
                        depth[r * w + col] = rd
                    continue
                v = (r - top) / max(line, 1e-6)
                uu = (u + (0.25 if int(v * 4) & 1 else 0.0)) % 0.5
                rgb[o : o + 3] = mortar if (v * 4) % 1.0 < 0.08 or uu < 0.04 else base
                if depth is not None:
                    depth[r * w + col] = dist
                    labels[r * w + col] = 1  # type: ignore[index]

        visible = []
        sprites = []
        for ent in self.entities:
            if not ent["alive"]:
                continue
            d, lateral = self._to_camera(ent)
            if d <= 0.15:
                continue
            screen_x = (w / 2) * (1 + lateral / (d * self.plane))
            sprites.append((d, lateral, screen_x, ent))
        for d, lateral, screen_x, ent in sorted(sprites, key=lambda s: -s[0]):
            style = ENTITY_STYLE[ent["kind"]]
            sh = h / d * style["size"]
            sw = sh * style["aspect"]
            bottom = horizon + h / (2 * d)
            top = bottom - sh
            x0, x1 = int(screen_x - sw / 2), int(screen_x + sw / 2)
            shown = False
            for col in range(max(0, x0), min(w, x1 + 1)):
                if d >= zbuf[col]:
                    continue
                shown = True
                for r in range(max(0, int(top)), min(h, int(bottom))):
                    v = (r - top) / max(sh, 1e-6)
                    color = style["accent"] if 0.15 < v < 0.25 else style["color"]
                    k = max(0.3, 1.0 - d / 12.0)
                    o = (r * w + col) * 3
                    rgb[o] = int(color[0] * k)
                    rgb[o + 1] = int(color[1] * k)
                    rgb[o + 2] = int(color[2] * k)
                    if depth is not None:
                        depth[r * w + col] = d
                        labels[r * w + col] = ent["id"]  # type: ignore[index]
            if shown:
                visible.append(
                    {
                        "id": ent["id"],
                        "kind": ent["kind"],
                        "bearing": round(math.degrees(math.atan2(lateral, d)), 2),
                        "distance": round(math.hypot(d, lateral), 3),
                    }
                )

        state = {
            "pos": [round(self.x, 4), round(self.y, 4)],
            "yaw": round(math.degrees(self.yaw), 3),
            "pitch": round(math.degrees(self.pitch), 3),
            "health": round(self.health, 1),
            "score": self.score,
            "deaths": self.deaths,
            "objective": self.objective,
            "visible": sorted(visible, key=lambda e: e["distance"]),
        }
        return EnvFrame(rgb=bytes(rgb), width=w, height=h, state=state, depth=depth, labels=labels)
