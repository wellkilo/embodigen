"""FidelityValidator: does the rendered video still support the action labels?

The validator only reads what a downstream consumer would have: the rendered
pixels plus camera intrinsics. It recovers the manipulated object and the
gripper from the rendered frames, projects the engine's ground-truth keypoints
through the *same* camera, and reports the reprojection error in millimetres
at the object's depth.

Reprojection is the right measurement here. Back-projecting a silhouette
centroid with the depth of its front surface recovers a point on the visible
face, not the object centre, which bakes a 2-3 cm bias into a 4 cm cube.
Reprojection compares like with like: "is the gripper rendered where the
engine put it", which is exactly the property that breaks when a
photorealizer drifts geometry near contact.

The gate is the end-effector error. It is the label that actually drives the
action, and it stays trackable while the manipulated object is occluded by
the fingers. Object-side error is reported alongside as a secondary number.

A production validator swaps colour segmentation for keypoint / pose
tracking or a learned inverse-dynamics model; the scoring contract and the
retry loop stay the same.
"""

from __future__ import annotations

import math

from embodigen.geometry import CameraModel
from embodigen.types import FidelityReport, FrameGT, PhotorealResult, SceneSpec

FINGER_CENTROID_OFFSET_Z = 0.018  # fingers sit above the grasp point

# Reference appearance per material class, as chromaticity directions only:
# shading multiplies all channels by the same factor, so direction survives
# lighting while absolute luminance does not. The cube borrows its own albedo;
# the rest are material constants of this scene family.
MATERIAL_REFS: dict[str, tuple[float, float, float]] = {
    "table": (0.66, 0.57, 0.47),
    "bin": (0.35, 0.65, 0.65),
    "gripper": (0.53, 0.58, 0.70),
}

MATCH_THRESHOLD = 0.16


def _ref_from_rgb(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (c / 255.0 for c in rgb)
    n = (r * r + g * g + b * b) ** 0.5 or 1.0
    return (r / n, g / n, b / n)


def _classify(
    r: int,
    g: int,
    b: int,
    refs: dict[str, tuple[float, float, float]],
) -> str | None:
    """Nearest reference in chromaticity space."""
    rr, gg, bb = r / 255.0, g / 255.0, b / 255.0
    n = (rr * rr + gg * gg + bb * bb) ** 0.5
    if n < 0.10:
        return None
    chroma = (rr / n, gg / n, bb / n)
    best, best_d = None, 1e9
    for name, ref_chroma in refs.items():
        dc = sum((chroma[i] - ref_chroma[i]) ** 2 for i in range(3)) ** 0.5
        if dc < best_d:
            best, best_d = name, dc
    return best if best_d < MATCH_THRESHOLD else None


class KeypointFidelityValidator:
    name = "keypoint-appearance-v0"

    def score(
        self,
        rendered: PhotorealResult,
        gt: list[FrameGT],
        scene: SceneSpec,
        threshold_mm: float,
    ) -> FidelityReport:
        cam = CameraModel(scene.camera)
        w, h = cam.camera.width, cam.camera.height
        px_to_mm = 1000.0 / cam.camera.fx
        refs = self._references(scene)

        per_frame: list[float] = []
        cube_errors: list[float] = []
        eef_errors: list[float] = []
        notes: list[str] = []
        cube_misses = 0
        eef_misses = 0

        for index, (frame, tgt) in enumerate(zip(rendered.frames, gt)):
            finger_px = self._centroid(frame.rgb, tgt.depth, w, h, refs, "gripper")
            cube_px = self._centroid(frame.rgb, tgt.depth, w, h, refs, "cube")

            eu, ev, ez = cam.project(
                (tgt.eef[0], tgt.eef[1], tgt.eef[2] + FINGER_CENTROID_OFFSET_Z)
            )
            if finger_px is None:
                eef_misses += 1
                per_frame.append(float("nan"))
            else:
                eef_err = math.hypot(finger_px[0] - eu, finger_px[1] - ev) * ez * px_to_mm
                eef_errors.append(eef_err)
                per_frame.append(eef_err)

            cu, cv, cz = cam.project(tgt.objects["cube"])
            if cube_px is None:
                cube_misses += 1
            else:
                cube_errors.append(
                    math.hypot(cube_px[0] - cu, cube_px[1] - cv) * cz * px_to_mm
                )

        if eef_misses:
            notes.append(f"gripper untracked in {eef_misses}/{len(gt)} frames")
        if cube_misses:
            notes.append(
                f"object untracked in {cube_misses}/{len(gt)} frames (occlusion / appearance)"
            )

        valid = [e for e in per_frame if e == e]
        if not valid:
            return FidelityReport(
                passed=False,
                threshold_mm=threshold_mm,
                mean_error_mm=float("inf"),
                max_error_mm=float("inf"),
                p95_error_mm=float("inf"),
                per_frame_mm=per_frame,
                notes=notes + ["end effector could not be tracked in any frame"],
            )

        ordered = sorted(valid)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        if cube_errors:
            notes.append(
                f"object mean {sum(cube_errors)/len(cube_errors):.1f} mm "
                f"max {max(cube_errors):.1f} mm (reported, not gated)"
            )
        notes.append(
            f"eef mean {sum(eef_errors)/len(eef_errors):.1f} mm, gate on eef p95"
        )

        return FidelityReport(
            passed=p95 <= threshold_mm,
            threshold_mm=threshold_mm,
            mean_error_mm=sum(valid) / len(valid),
            max_error_mm=max(valid),
            p95_error_mm=p95,
            per_frame_mm=per_frame,
            notes=notes,
        )

    def _references(self, scene: SceneSpec) -> dict[str, tuple[float, float, float]]:
        refs = dict(MATERIAL_REFS)
        try:
            refs["cube"] = _ref_from_rgb(scene.object("cube").albedo)
        except KeyError:
            pass
        return refs

    @staticmethod
    def _centroid(
        rgb: bytes,
        depth: list[float],
        w: int,
        h: int,
        refs: dict[str, tuple[float, float, float]],
        label: str,
    ) -> tuple[float, float] | None:
        far = max(depth) if depth else 0.0
        total = 0
        sx = 0.0
        sy = 0.0
        for v in range(h):
            row = v * w
            for u in range(w):
                idx = row + u
                if depth[idx] >= far - 1e-6:  # background: no geometry, no cue
                    continue
                p = idx * 3
                if _classify(rgb[p], rgb[p + 1], rgb[p + 2], refs) == label:
                    sx += u
                    sy += v
                    total += 1
        if total < 6:
            return None
        return (sx / total, sy / total)
