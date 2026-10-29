"""Gameplay -> embodied: certified recordings as a LeRobot-shaped dataset.

``build`` writes the GameScaling training format: chat samples where the
model sees a frame and answers with the action as text. Embodied-policy code
(LeRobot, most VLA stacks) reads episodes instead: one row per control step
with an observation image, a numeric state and a numeric action vector. This
module writes the certified recordings of a curated pool in that shape, one
episode per recording, one row per 200 ms action packet::

    <out>/
    ├── meta/info.json        features, fps, splits, path templates (LeRobot v2.1 keys)
    ├── meta/episodes.jsonl   episode_index, tasks, length, and the recording it came from
    ├── meta/tasks.jsonl      task_index -> task text
    ├── meta/stats.json       min / max / mean / std per numeric feature (for normalisation)
    ├── data/chunk-000/episode_000000.jsonl   (+ .parquet when pyarrow is installed)
    └── images/
        ├── observation.images.main/episode_000000/frame_000000.png
        ├── observation.depth/episode_000000/frame_000000.f32      engine rollouts only
        └── observation.segmentation/episode_000000/frame_000000.u16

``action`` (float32, 3 + len(KEY_ORDER) + 1), names in ``info.json``:

* ``mouse_dx, mouse_dy, wheel``: the canonical packet in counts (|dx|, |dy| <= 1000,
  |wheel| <= 5), after the source's mouse scale and any realignment;
* ``key.<name>`` for every key in ``actions.KEY_ORDER``: the fraction of the six 33 ms
  slots in which it is held (0, 1/6, ... 1);
* ``key.other``: the fraction of slots holding any key outside that list.

The vector drops the order of keys inside the 200 ms; the row keeps it, and the
exact GameScaling text: ``action.keys`` (six slots) and ``action.text``
(``encode_action``). The labels are the curated turns, unchanged.

``observation.state`` (float32: x, y, z, yaw, pitch; z up, each game's own
units and angle convention) is written only when every exported recording
has a pose on every step (engine rollouts, VPT). Otherwise the feature is left
out of ``info.json``; whatever state a step has stays in
``observation.state_raw``.

Frames kept in videos (VPT, D2E, P2P) are decoded to PNG at the original
capture aspect, which needs the ``[video]`` extra; the run checks that before
writing anything.
"""

from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from embodigen.gameplay.actions import CHUNK_S, KEY_ORDER, OBS_HZ, SUBSTEPS, encode_action, turn_from_json
from embodigen.gameplay.pipeline import _jsonl, _write_json
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.splits import session_of
from embodigen.gameplay.video import capture_aspect, fit_size, split_ref, video_reader
from embodigen.pngio import encode_png, read_png

FORMAT = "embodigen-lerobot-shaped-v1"  # keys follow LeRobot v2.1; not loadable by LeRobotDataset as is
CHUNKS_SIZE = 1000
DECODE_BATCH = 64
IMAGE = "observation.images.main"
DEPTH = "observation.depth"
SEG = "observation.segmentation"
ACTION_NAMES = ["mouse_dx", "mouse_dy", "wheel", *(f"key.{k}" for k in KEY_ORDER), "key.other"]
STATE_NAMES = ["x", "y", "z", "yaw", "pitch"]
_KEY_INDEX = {k: 3 + i for i, k in enumerate(KEY_ORDER)}
_OTHER = len(ACTION_NAMES) - 1


def action_vector(dx: int, dy: int, dz: int, keys: list[list[str]]) -> list[float]:
    vec = [float(dx), float(dy), float(dz)] + [0.0] * (len(ACTION_NAMES) - 3)
    for slot in keys:
        known = {k for k in slot if k in _KEY_INDEX}
        for key in known:
            vec[_KEY_INDEX[key]] += 1.0 / SUBSTEPS
        if len(known) < len(set(slot)):  # once per slot, however many unlisted keys
            vec[_OTHER] += 1.0 / SUBSTEPS
    return [round(v, 6) for v in vec]


def pose(state: dict[str, Any] | None) -> list[float] | None:
    """x, y, z (up), yaw, pitch from a step's state, or None if it has no full pose."""
    if not state:
        return None
    if "pose" in state:  # VPT: xpos, ypos (height), zpos, yaw, pitch
        p = state["pose"]
        if len(p) == 5 and all(isinstance(v, (int, float)) for v in p):
            return [float(p[0]), float(p[2]), float(p[1]), float(p[3]), float(p[4])]
        return None
    if "pos" in state and "yaw" in state:  # engine rollouts: a plane, z = 0
        x, y = state["pos"][:2]
        return [float(x), float(y), 0.0, float(state["yaw"]), float(state.get("pitch", 0.0))]
    return None


def _png_header(path: Path) -> tuple[int, int, bool]:
    """(width, height, whether the pixels are already 8-bit RGB) from the IHDR chunk."""
    with path.open("rb") as fh:
        head = fh.read(26)
    rgb8 = head[24] == 8 and head[25] == 2  # bit depth 8, colour type 2
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"), rgb8


def letterbox(w: int, h: int, rgb: bytes, width: int, height: int) -> bytes:
    """Scale packed RGB (nearest neighbour) to fit width x height at its own aspect,
    centred on black bars, so every frame of the dataset has one shape."""
    fw, fh = fit_size(w / h, width, height)
    if (fw, fh) != (w, h):
        xs = [min(w - 1, x * w // fw) * 3 for x in range(fw)]
        rows = [rgb[y * w * 3 : (y + 1) * w * 3] for y in range(h)]
        rgb = b"".join(b"".join(row[o : o + 3] for o in xs) for row in (rows[min(h - 1, y * h // fh)] for y in range(fh)))
    if (fw, fh) == (width, height):
        return rgb
    left, top = (width - fw) // 2, (height - fh) // 2
    pad_l, pad_r = b"\0" * (left * 3), b"\0" * ((width - fw - left) * 3)
    blank = b"\0" * (width * 3)
    body = b"".join(pad_l + rgb[y * fw * 3 : (y + 1) * fw * 3] + pad_r for y in range(fh))
    return blank * top + body + blank * (height - fh - top)


def _task(world_title: str, state: dict[str, Any] | None) -> str:
    """A step's language task: the source's own instruction when it has one (P2P),
    else what the player is doing in which game."""
    if state:
        if state.get("instruction"):
            return str(state["instruction"])
        if state.get("objective"):
            return f"Play {world_title}: {state['objective']}"
    return f"Play {world_title}"


def plan(curated: Path, worlds: set[str] | None = None) -> tuple[Path, list[dict[str, Any]]]:
    """(raw root, certified recording rows to export), in a stable order."""
    ledger_path = curated / "ledger.json"
    if not ledger_path.exists():
        raise SystemExit(f"{curated} has no ledger.json; run `embodigen curate` first")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    raw_root = Path(ledger["raw_root"])
    rows = [r for r in _jsonl(curated / "recordings.jsonl") if r["passed"]]
    if worlds:
        unknown = worlds - {r["world"] for r in rows}
        if unknown:
            raise SystemExit(f"no certified recordings for {', '.join(sorted(unknown))} in {curated}")
        rows = [r for r in rows if r["world"] in worlds]
    if not rows:
        raise SystemExit(f"no certified recordings in {curated}; nothing to export")
    # Same rule as splits.split: a source's "val" tag holds out its whole session.
    # Held-out recordings go last, so each split is one contiguous episode range.
    held = {session_of(r) for r in rows if r.get("split_hint") == "val"}
    for r in rows:
        r["_split"] = "val" if session_of(r) in held else "train"
    rows.sort(key=lambda r: (r["_split"] == "val", r["recording_id"]))
    return raw_root, rows


def _needs_video(curated: Path, rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows:
        with (curated / row["turns_file"]).open(encoding="utf-8") as fh:
            first = fh.readline()
        if first and split_ref(json.loads(first)["frame"])[1] is not None:
            out.append(row["recording_id"])
    return out


def _episode(job: dict[str, Any]) -> dict[str, Any]:
    """Write one episode's media and return its rows (indexed later). Runs in a worker process."""
    out, curated, raw = Path(job["out"]), Path(job["curated"]), Path(job["raw"])
    ep, row, title = job["episode_index"], job["row"], job["title"]
    turns_json = _jsonl(curated / row["turns_file"])
    turns = [turn_from_json(t) for t in turns_json]
    obs = {o["frame"]: o for o in _jsonl(raw / "obs.jsonl")} if (raw / "obs.jsonl").exists() else {}

    img_dir = out / "images" / IMAGE / f"episode_{ep:06d}"
    img_dir.mkdir(parents=True, exist_ok=True)
    video = [i for i, t in enumerate(turns) if split_ref(t.frame)[1] is not None]
    if video:
        reader, why = video_reader(job["video_backend"])
        if reader is None:
            raise SystemExit(f"{row['recording_id']}: frames are inside a video and {why}")
        aspect = capture_aspect(raw)
        # Small batches: a decoder's frame pool runs dry (EAGAIN) if thousands of
        # decoded frames are held at once, and memory would grow with the recording.
        W, H = job["size"]
        for b in range(0, len(video), DECODE_BATCH):
            idx = video[b : b + DECODE_BATCH]
            for i, (w, h, rgb) in zip(idx, reader.rgbs(raw, [turns[i].frame for i in idx], W, aspect, fit=(W, H))):
                (img_dir / f"frame_{i:06d}.png").write_bytes(encode_png(W, H, letterbox(w, h, rgb, W, H)))

    gt = {DEPTH: [], SEG: []}
    rows_out, states = [], []
    for i, (turn, tj) in enumerate(zip(turns, turns_json)):
        name = f"frame_{i:06d}"
        image = img_dir / f"{name}.png"
        if split_ref(turn.frame)[1] is None:
            W, H = job["size"]
            if _png_header(raw / turn.frame) == (W, H, True):
                shutil.copyfile(raw / turn.frame, image)
            else:
                w, h, c, px = read_png(raw / turn.frame)
                if c != 3:
                    px = b"".join(px[k : k + 3] for k in range(0, len(px), c)) if c == 4 else bytes(v for v in px for _ in range(3))
                image.write_bytes(encode_png(W, H, letterbox(w, h, px, W, H)))
        o = obs.get(turn.frame, {})
        frames = {IMAGE: image.relative_to(out).as_posix()}
        for feature, key, ext in ((DEPTH, "depth", ".f32"), (SEG, "labels", ".u16")):
            if o.get(key) and (raw / o[key]).exists() and job["native"]:
                dst = out / "images" / feature / f"episode_{ep:06d}" / f"{name}{ext}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw / o[key], dst)
                frames[feature] = dst.relative_to(out).as_posix()
                gt[feature].append(True)
            else:
                gt[feature].append(False)
        state = tj.get("state")
        p = pose(state)
        states.append(p)
        rows_out.append({
            "episode_index": ep,
            "frame_index": i,
            "timestamp": round(i * CHUNK_S, 6),
            "source_time": turn.t,
            "task": _task(title, state),
            "frames": frames,
            "action": action_vector(turn.action.dx, turn.action.dy, turn.action.dz, tj["keys"]),
            "action.keys": tj["keys"],
            "action.text": encode_action(turn.action),
            "observation.state": p,
            "observation.state_raw": state or {},
            "next.done": i == len(turns) - 1,
        })
    return {
        "episode_index": ep, "rows": rows_out,
        "has_pose": bool(states) and all(s is not None for s in states),
        "depth": bool(gt[DEPTH]) and all(gt[DEPTH]), "seg": bool(gt[SEG]) and all(gt[SEG]),
    }


def _stats(vectors: list[list[float]]) -> dict[str, list[float]]:
    n = len(vectors)
    cols = list(zip(*vectors))
    mean = [sum(c) / n for c in cols]
    std = [math.sqrt(sum((v - m) ** 2 for v in c) / n) for c, m in zip(cols, mean)]
    return {"min": [min(c) for c in cols], "max": [max(c) for c in cols],
            "mean": [round(m, 6) for m in mean], "std": [round(s, 6) for s in std], "count": [n]}


def _maybe_parquet(out: Path, ep: int, rows: list[dict[str, Any]], with_state: bool) -> bool:
    try:
        import pyarrow as pa  # type: ignore  # the [lerobot] extra
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return False
    table = pa.table({
        "episode_index": [r["episode_index"] for r in rows],
        "frame_index": [r["frame_index"] for r in rows],
        "index": [r["index"] for r in rows],
        "task_index": [r["task_index"] for r in rows],
        "timestamp": [r["timestamp"] for r in rows],
        IMAGE: [r["frames"][IMAGE] for r in rows],
        "action": [r["action"] for r in rows],
        **({"observation.state": [r["observation.state"] for r in rows]} if with_state else {}),
        "next.done": [r["next.done"] for r in rows],
    })
    pq.write_table(table, out / "data" / f"chunk-{ep // CHUNKS_SIZE:03d}" / f"episode_{ep:06d}.parquet")
    return True


def frame_size(curated: Path, raw_root: Path, rows: list[dict[str, Any]], need_video: list[str],
               width: int) -> tuple[tuple[int, int], bool]:
    """One frame size for the whole dataset, and whether it is every frame's own.

    The engine frames' size when no video is involved and they all agree; else
    ``width`` at 16:9 (the capture shape of every public dataset here), with
    other frames letterboxed into it."""
    if not need_video:
        sizes = set()
        for row in rows:
            with (curated / row["turns_file"]).open(encoding="utf-8") as fh:
                frame = json.loads(fh.readline())["frame"]
            sizes.add(_png_header(raw_root / row["raw_dir"] / frame)[:2])
        if len(sizes) == 1:
            return sizes.pop(), True
    return (width, round(width * 9 / 16)), False


def export(curated: Path, out: Path, worlds: set[str] | None = None, width: int = 320,
           video_backend: str = "pyav", workers: int = 1, overwrite: bool = False,
           registry: WorldRegistry | None = None, size: tuple[int, int] | None = None) -> dict[str, Any]:
    """``size`` (w, h) forces every frame to that shape; by default see ``frame_size``."""
    curated, out = Path(curated), Path(out)
    raw_root, rows = plan(curated, worlds)
    need = _needs_video(curated, rows)
    if need:  # fail before writing anything
        reader, why = video_reader(video_backend)
        if reader is None:
            raise SystemExit(
                f"{len(need)} recording(s) keep their frames inside videos ({', '.join(need[:3])}"
                f"{', ...' if len(need) > 3 else ''}): {why}\n"
                "or export only engine worlds with --worlds.")
    replace = out.exists() and any(out.iterdir())
    if replace:
        if not overwrite:
            raise SystemExit(f"{out} is not empty; pass --overwrite to replace an earlier export")
        try:
            ours = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8")).get("format") == FORMAT
        except (OSError, ValueError):
            ours = False
        if not ours:  # e.g. a downloaded LeRobot dataset, which also has meta/info.json
            raise SystemExit(f"{out} is not empty and was not written by `embodigen export`; refusing to delete it")
    # Build beside the target and swap at the end: a failed run leaves the old export intact.
    final, out = out, out.parent / f".{out.name}.partial"
    if out.exists():
        shutil.rmtree(out)  # leftovers of a crashed run, by this name only ours
    out.mkdir(parents=True)
    try:
        result = _write(curated, out, raw_root, rows, need, width, size, video_backend, workers, registry)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final) if replace else final.rmdir()
    out.rename(final)
    return {**result, "out": str(final)}


def _write(curated: Path, out: Path, raw_root: Path, rows: list[dict[str, Any]], need: list[str], width: int,
           size: tuple[int, int] | None, video_backend: str, workers: int,
           registry: WorldRegistry | None) -> dict[str, Any]:
    registry = registry or WorldRegistry.load()
    auto, uniform = frame_size(curated, raw_root, rows, need, width)
    size = size or auto
    native = uniform and size == auto  # every engine frame copied as is, so depth/seg pixels line up
    jobs = [{"out": str(out), "curated": str(curated), "raw": str(raw_root / row["raw_dir"]), "row": row,
             "episode_index": ep, "title": registry.get(row["world"]).title, "size": size,
             "native": native, "video_backend": video_backend} for ep, row in enumerate(rows)]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            done = list(pool.map(_episode, jobs))
    else:
        done = [_episode(job) for job in jobs]

    with_state = all(d["has_pose"] for d in done)
    with_depth = all(d["depth"] for d in done)
    with_seg = all(d["seg"] for d in done)
    tasks: dict[str, int] = {}
    total, episodes, actions, states, parquet = 0, [], [], [], False
    for d, row in zip(done, rows):
        for r in d["rows"]:
            r["task_index"] = tasks.setdefault(r["task"], len(tasks))
            r["index"] = total
            total += 1
            actions.append(r["action"])
            if with_state:
                states.append(r["observation.state"])
        path = out / "data" / f"chunk-{d['episode_index'] // CHUNKS_SIZE:03d}" / f"episode_{d['episode_index']:06d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in d["rows"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        parquet = _maybe_parquet(out, d["episode_index"], d["rows"], with_state) or parquet
        episodes.append({
            "episode_index": d["episode_index"],
            "tasks": list(dict.fromkeys(r["task"] for r in d["rows"])),
            "length": len(d["rows"]),
            "recording_id": row["recording_id"], "world": row["world"], "source": row["source"],
            "session": session_of(row), "split": row["_split"],
            "realigned_ms": row.get("realigned_ms", 0.0),
        })

    n_train = sum(1 for e in episodes if e["split"] == "train")
    splits = {"train": f"0:{n_train}"}
    if n_train < len(episodes):
        splits["val"] = f"{n_train}:{len(episodes)}"
    w, h = size
    features: dict[str, Any] = {
        IMAGE: {"dtype": "image", "shape": [h, w, 3], "names": ["height", "width", "channel"],
                "info": "PNG; frames of another aspect are letterboxed (black bars), never stretched"},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    if with_state:
        features["observation.state"] = {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES}
    if with_depth:
        features[DEPTH] = {"dtype": "float32", "shape": [h, w], "names": ["height", "width"],
                           "info": "raw little-endian float32 metres, engine truth (.f32)"}
    if with_seg:
        features[SEG] = {"dtype": "uint16", "shape": [h, w], "names": ["height", "width"],
                         "info": "raw little-endian uint16 instance ids, engine truth (.u16)"}
    info = {
        "codebase_version": FORMAT,  # not "v2.1": LeRobotDataset should refuse it rather than misread it
        "format": FORMAT,
        "robot_type": "keyboard_mouse",
        "fps": int(OBS_HZ),
        "total_episodes": len(episodes),
        "total_frames": total,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (len(episodes) - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "splits": splits,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}." + ("parquet" if parquet else "jsonl"),
        "image_path": "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
        "video_path": None,
        "features": features,
        "source": {"curated": str(curated.resolve()), "raw_root": str(raw_root),
                   "worlds": sorted({e["world"] for e in episodes})},
        "note": ("One row per 200 ms GameScaling action packet: the frame at its start and the keyboard/mouse "
                 "action during it. Keys follow LeRobot v2.1, but frames are PNG files referenced by path, rows "
                 "are JSONL (parquet only with pyarrow) and there is no episodes_stats.jsonl, so LeRobotDataset "
                 "needs a conversion first. action.keys / action.text keep the exact slots the vector summarises."),
    }
    _write_json(out / "meta" / "info.json", info)
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for task, i in tasks.items():
            fh.write(json.dumps({"task_index": i, "task": task}, ensure_ascii=False) + "\n")
    stats = {"action": _stats(actions)}
    if with_state:
        stats["observation.state"] = _stats(states)
    _write_json(out / "meta" / "stats.json", stats)
    return {"out": str(out), "episodes": len(episodes), "size": [w, h], "frames": total, "tasks": len(tasks),
            "splits": splits, "state": with_state, "depth": with_depth, "segmentation": with_seg,
            "parquet": parquet, "worlds": info["source"]["worlds"],
            "minutes": round(total * CHUNK_S / 60, 1)}
