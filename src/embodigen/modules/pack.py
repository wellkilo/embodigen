"""DatasetPacker: episode -> on-disk dataset.

Writes a LeRobot-shaped layout (``meta/`` + ``data/chunk-000/`` + frames) that
a consumer can read without importing this package. Frames are stored twice:
PNG for humans, raw ``.rgb``/``.f32``/``.u16`` buffers for anything that wants
exact values without a decoder.

Parquet is intentionally left to the ``[lerobot]`` extra; if ``pyarrow`` is
installed the rows are additionally written as parquet.
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

from embodigen.pngio import write_png
from embodigen.types import (
    Camera,
    Episode,
    FrameGT,
    ObjectSpec,
    PhotorealFrame,
    PhotorealResult,
    SceneSpec,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class LeRobotPacker:
    name = "lerobot-v0"

    def pack(self, episode: Episode, out_dir: Path) -> Path:
        root = Path(out_dir) / episode.episode_id
        if root.is_dir():
            # A re-run may produce fewer frames; never leave orphans behind.
            shutil.rmtree(root)
        for sub in ("meta", "data/chunk-000", "frames/photoreal", "frames/white", "frames/depth", "frames/seg"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        cam = episode.scene.camera
        w, h = cam.width, cam.height

        rows = []
        for i, (pframe, gt) in enumerate(zip(episode.photoreal.frames, episode.gt)):
            stem = f"{i:03d}"
            write_png(root / "frames/photoreal" / f"{stem}.png", w, h, pframe.rgb)
            (root / "frames/photoreal" / f"{stem}.rgb").write_bytes(pframe.rgb)
            write_png(root / "frames/white" / f"{stem}.png", w, h, gt.rgb)
            (root / "frames/white" / f"{stem}.rgb").write_bytes(gt.rgb)
            (root / "frames/depth" / f"{stem}.f32").write_bytes(
                struct.pack(f"<{w*h}f", *gt.depth)
            )
            (root / "frames/seg" / f"{stem}.u16").write_bytes(
                struct.pack(f"<{w*h}H", *gt.seg)
            )
            rows.append(
                {
                    "frame_index": i,
                    "timestamp": round(gt.t, 4),
                    "language_instruction": episode.task.instruction,
                    "frames": {
                        "photoreal": f"frames/photoreal/{stem}.png",
                        "white_model": f"frames/white/{stem}.png",
                        "depth": f"frames/depth/{stem}.f32",
                        "segmentation": f"frames/seg/{stem}.u16",
                    },
                    # Labels keep micron precision: enough to round-trip a
                    # re-validation, short enough to read in a JSONL row.
                    "action": [round(v, 6) for v in gt.action],
                    "observation": {
                        "eef_position": [round(v, 6) for v in gt.eef],
                        "gripper_width": round(gt.gripper, 6),
                        "object_positions": {
                            name: [round(c, 6) for c in pos]
                            for name, pos in gt.objects.items()
                        },
                    },
                    "fidelity": {"drift_px": pframe.drift_px},
                }
            )

        _write_json(
            root / "meta/info.json",
            {
                "codebase_version": "v0.1-mock",
                "robot_type": episode.scene.robot,
                "morphology": episode.scene.morphology,
                "fps": episode.trajectory.fps,
                "total_episodes": 1,
                "total_frames": len(rows),
                "features": episode.feature_summary(),
                "note": (
                    "LeRobot-shaped layout produced by the reference implementation. "
                    "Video encoding and parquet output require the [lerobot] extra."
                ),
            },
        )
        _write_json(root / "meta/provenance.json", asdict(episode.provenance))
        _write_json(
            root / "meta/fidelity.json",
            {
                **episode.fidelity.to_dict(),
                "validator": episode.provenance.validator or "unknown",
                "note": "errors measured in 3D from rendered pixels only",
            },
        )
        _write_json(
            root / "meta/scene.json",
            {
                "instruction": episode.scene.instruction,
                "template": episode.scene.template,
                "camera": episode.scene.camera.to_dict(),
                "objects": [asdict(o) for o in episode.scene.objects],
                "success": episode.scene.success,
                "notes": episode.scene.notes,
                "intrinsics": [cam.fx, cam.fy, cam.cx, cam.cy],
            },
        )
        with (root / "data/chunk-000/episode_000000.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._maybe_parquet(root, rows)
        return root

    @staticmethod
    def _maybe_parquet(root: Path, rows: list[dict[str, Any]]) -> None:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            return
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, root / "data/chunk-000/episode_000000.parquet")


def read_episode(root: str | Path) -> tuple[SceneSpec, list[FrameGT], PhotorealResult]:
    """Read a packed episode back into the pipeline types (for `embodigen validate`)."""
    root = Path(root)
    if not (root / "meta/scene.json").exists():
        raise SystemExit(f"{root} is not a packed episode (missing meta/scene.json)")
    scene_meta = json.loads((root / "meta/scene.json").read_text(encoding="utf-8"))
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    cam = Camera(**{k: tuple(v) if isinstance(v, list) else v for k, v in scene_meta["camera"].items()})
    objects = [
        ObjectSpec(
            name=o["name"],
            kind=o["kind"],
            center=tuple(o["center"]),
            half_extent=tuple(o["half_extent"]),
            material=o["material"],
            albedo=tuple(o["albedo"]),
            movable=o.get("movable", True),
            is_container=o.get("is_container", False),
        )
        for o in scene_meta["objects"]
    ]
    scene = SceneSpec(
        scene_id=root.name,
        instruction=scene_meta["instruction"],
        template=scene_meta["template"],
        morphology=info.get("morphology", "arm"),
        robot=info.get("robot_type", "franka"),
        camera=cam,
        objects=objects,
        success=scene_meta.get("success", {}),
        seed=0,
        notes=scene_meta.get("notes", []),
    )

    rows = [
        json.loads(line)
        for line in (root / "data/chunk-000/episode_000000.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    w, h = cam.width, cam.height
    gt: list[FrameGT] = []
    pframes: list[PhotorealFrame] = []
    for row in rows:
        i = row["frame_index"]
        depth = list(struct.unpack(f"<{w*h}f", (root / f"frames/depth/{i:03d}.f32").read_bytes()))
        seg = list(struct.unpack(f"<{w*h}H", (root / f"frames/seg/{i:03d}.u16").read_bytes()))
        white = (root / f"frames/white/{i:03d}.rgb").read_bytes() if (root / f"frames/white/{i:03d}.rgb").exists() else b""
        gt.append(
            FrameGT(
                t=row["timestamp"],
                rgb=white,
                depth=depth,
                seg=seg,
                eef=tuple(row["observation"]["eef_position"]),
                gripper=row["observation"]["gripper_width"],
                objects={
                    name: tuple(pos)
                    for name, pos in row["observation"]["object_positions"].items()
                },
                action=tuple(row["action"]),
            )
        )
        pframes.append(
            PhotorealFrame(
                t=row["timestamp"],
                rgb=(root / f"frames/photoreal/{i:03d}.rgb").read_bytes(),
                drift_px=row.get("fidelity", {}).get("drift_px", 0.0),
            )
        )
    return scene, gt, PhotorealResult(model="from-disk", seed=0, frames=pframes)
