"""Photorealizer: white-model frames -> photoreal-looking frames.

``MockPhotorealizer`` is a deterministic stand-in for the fine-tuned Wan
model. It keeps the same interface (frames + scene -> frames) and the same
failure mode that matters: it can drift the geometry of the manipulated
object near contact, which is exactly what the validator must catch.

It deliberately does **not** claim photorealism. What it demonstrates is the
data path and the closed loop, so the contract can be nailed down before a
GPU-scale renderer lands.
"""

from __future__ import annotations

import math

from embodigen.geometry import CameraModel, normalize
from embodigen.types import FrameGT, PhotorealFrame, PhotorealResult, SceneSpec

STUDIO_TOP = (52, 52, 54)
STUDIO_BOTTOM = (20, 20, 22)


def _hash_noise(x: int, y: int, seed: int) -> float:
    h = (x * 374761393 + y * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


class MockPhotorealizer:
    name = "mock-photoreal-v0"

    def __init__(
        self,
        drift_mm: float = 0.0,
        drift_frames: tuple[int, int] = (0, 0),
        drift_px: float | None = None,
        grain: float = 1.0,
    ) -> None:
        # Drift is specified in millimetres of world-space displacement, so the
        # failure it simulates is the same at every render resolution. ``drift_px``
        # remains for tests that want a fixed pixel offset.
        self.drift_mm = drift_mm
        self.drift_px = drift_px
        self.drift_frames = drift_frames
        # Film grain is part of the look, but it is per-pixel noise: at small
        # sizes with a 256-colour palette it turns into confetti, so callers
        # (GIF export) can dial it down.
        self.grain = grain

    def render(
        self,
        gt: list[FrameGT],
        scene: SceneSpec,
        condition_weights: dict[str, float],
        seed: int,
    ) -> PhotorealResult:
        cam = CameraModel(scene.camera)
        palette = self._palette(scene)
        frames: list[PhotorealFrame] = []
        lo, hi = self.drift_frames

        for index, frame in enumerate(gt):
            drift_px = self._resolve_drift(frame, cam)
            drifting = drift_px > 0 and lo <= index <= hi
            rgb = self._shade_frame(frame, cam, palette, scene, seed, drift_px if drifting else 0)
            frames.append(
                PhotorealFrame(t=frame.t, rgb=rgb, drift_px=drift_px if drifting else 0.0)
            )

        return PhotorealResult(
            model=self.name,
            seed=seed,
            frames=frames,
            condition_weights=dict(condition_weights),
        )

    def _resolve_drift(self, frame: FrameGT, cam: CameraModel) -> float:
        if self.drift_px is not None:
            return float(self.drift_px)
        if self.drift_mm <= 0:
            return 0.0
        z = cam.project(frame.eef)[2]
        return self.drift_mm / 1000.0 * cam.camera.fx / max(z, 0.05)

    def _palette(self, scene: SceneSpec) -> dict[str, tuple[float, float, float]]:
        """Material appearance per instance name (photoreal albedo, not clay)."""
        out: dict[str, tuple[float, float, float]] = {}
        for obj in scene.objects:
            if obj.material == "table":
                out[obj.name] = (0.58, 0.50, 0.41)
            elif obj.material == "bin":
                out[obj.name] = (0.14, 0.26, 0.26)
            elif obj.material == "cube":
                out[obj.name] = tuple(c / 255.0 for c in obj.albedo)  # type: ignore[assignment]
            elif obj.material == "gripper":
                out[obj.name] = (0.62, 0.68, 0.82)
            else:
                out[obj.name] = tuple(c / 255.0 for c in obj.albedo)  # type: ignore[assignment]
        return out

    def _shade_frame(
        self,
        frame: FrameGT,
        cam: CameraModel,
        palette: dict[str, tuple[float, float, float]],
        scene: SceneSpec,
        seed: int,
        drift_px: float,
    ) -> bytes:
        w, h = cam.camera.width, cam.camera.height
        depth = list(frame.depth)
        seg = list(frame.seg)
        name_of = {v: k for k, v in frame.seg_labels.items()}
        out = bytearray(w * h * 3)

        moving = {label for name, label in frame.seg_labels.items() if "finger" in name or name == "cube"}
        drift = int(round(drift_px))

        if drift:
            # Rigid, local displacement of the manipulated geometry: clear the
            # original footprint to the table, then stamp it back shifted. The
            # vacated area is what a drifted render would actually reveal.
            table_id = frame.seg_labels.get("table", 0)
            source_seg, source_depth = list(seg), list(depth)
            shifted_seg, shifted_depth = list(seg), list(depth)
            for v in range(h):
                for u in range(w):
                    p = v * w + u
                    if source_seg[p] in moving:
                        shifted_seg[p] = table_id
                        shifted_depth[p] = source_depth[p] + 0.004
            for v in range(h):
                for u in range(w):
                    p = v * w + u
                    if source_seg[p] not in moving:
                        continue
                    # Horizontal-only shift keeps "N px" a clean proxy for
                    # millimetres of world displacement (no sqrt(2) factor).
                    du = u + drift
                    dv = v
                    if 0 <= du < w and 0 <= dv < h:
                        q = dv * w + du
                        shifted_seg[q] = source_seg[p]
                        shifted_depth[q] = source_depth[p]
            seg, depth = shifted_seg, shifted_depth

        for v in range(h):
            for u in range(w):
                p = v * w + u
                sp = p
                sid = seg[sp]
                z = depth[sp]

                if sid == 0:
                    k = v / max(1, h - 1)
                    r = STUDIO_TOP[0] + (STUDIO_BOTTOM[0] - STUDIO_TOP[0]) * k
                    g = STUDIO_TOP[1] + (STUDIO_BOTTOM[1] - STUDIO_TOP[1]) * k
                    b = STUDIO_TOP[2] + (STUDIO_BOTTOM[2] - STUDIO_TOP[2]) * k
                    out[p * 3], out[p * 3 + 1], out[p * 3 + 2] = int(r), int(g), int(b)
                    continue

                name = name_of.get(sid, "table")
                base = palette.get(name, (0.5, 0.5, 0.5))

                # Normals from the depth buffer: cheap but geometry-aware.
                zr = depth[v * w + min(w - 1, u + 1)]
                zd = depth[min(h - 1, v + 1) * w + u]
                # Guard against depth discontinuities across silhouettes.
                if abs(zr - z) > 0.05:
                    zr = z
                if abs(zd - z) > 0.05:
                    zd = z
                n = normalize((-(zr - z) * 28.0, -(zd - z) * 28.0, 1.0))
                lam = max(0.0, n[0] * -0.45 + n[1] * -0.5 + n[2] * 0.74)
                spec = max(0.0, lam) ** 12 * 0.12 if name.startswith("cube") else 0.0
                ao = 0.72 + 0.28 * min(1.0, max(0.0, (1.25 - z) / 0.7))
                shade = 0.22 + 0.95 * lam

                grain = (_hash_noise(u, v, seed) - 0.5) * 0.045 * self.grain
                texture = 1.0
                if name == "table":
                    texture = (
                        1.0
                        + 0.028 * math.sin(u * 0.42 + seed * 0.7)
                        + 0.012 * math.sin(v * 0.17 + u * 0.05)
                    )
                elif name == "cube":
                    texture = 1.0 + 0.03 * math.sin((u + v) * 0.8)

                k = shade * ao * texture + grain
                for c in range(3):
                    vv = base[c] * k + spec
                    vv = vv * (1.0 - 0.18 * ((u - w / 2) ** 2 + (v - h / 2) ** 2) / (w * h))
                    out[p * 3 + c] = max(0, min(255, int(vv * 255)))

        return bytes(out)
