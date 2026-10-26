"""Orchestration for the virtual-world track.

    rollout / ingest  ->  raw/<recording>/            (engine plays, or a dataset converts)
    curate            ->  curated/turns + recordings  (align, canonicalise, gate)
    build             ->  mixes/<study>/...           (split, window, mix, annotate, shard)

Each stage reads only the previous stage's directory, so they can run on
different machines, and each writes a ledger that says what it kept, what it
dropped and why. As in the robot track, a recording that fails the gate is
never silently used: it is listed in ``curated/rejected.jsonl`` and has no
turns file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodigen.gameplay.actions import Canonicalizer, turn_from_json, turn_to_json
from embodigen.gameplay.annotate import DEFAULT_RATE, RationaleCache, make_annotator, selected
from embodigen.gameplay.mixture import MixtureBuilder, StudySpec, diversity
from embodigen.gameplay.quality import AlignmentChecker, QualityConfig, assess, check_alignment
from embodigen.gameplay.recording import discover, read_recording
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.rollout import RolloutSpec, rollout
from embodigen.gameplay.serialize import ShardWriter, write_samples
from embodigen.gameplay.splits import SplitConfig, split
from embodigen.gameplay.types import Turn, Window
from embodigen.gameplay.windows import WindowConfig, cap_streams, downsample_idle, footage_seconds, make_windows
from embodigen.modules.base import resolve


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pool(workers: int, n: int):
    return ProcessPoolExecutor(max_workers=min(workers, n)) if workers > 1 and n > 1 else None


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# -- rollout --------------------------------------------------------------------

def _rollout_one(spec: RolloutSpec, out_root: str, recording_id: str) -> dict[str, Any]:
    env = resolve("env", spec.engine)(width=spec.width, height=spec.height, **spec.engine_args)
    try:
        meta = rollout(env, spec, Path(out_root), recording_id)
    finally:
        env.close()
    return {"recording_id": meta.recording_id, "world": meta.world, "duration": meta.duration,
            "stream": meta.stream, "lag_ms": spec.lag_ms, "profile": spec.profile}


def run_rollouts(
    registry: WorldRegistry,
    worlds: list[str],
    episodes: int,
    seconds: float,
    out_root: Path,
    seed: int = 0,
    profile: str = "player",
    lag_ms: float = 0.0,
    drift_ms_per_min: float = 0.0,
    size: tuple[int, int] = (160, 100),
    streams: int = 2,
    gt: bool = True,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """``episodes`` recordings per world on consecutive seeds."""
    jobs = []
    for world in worlds:
        spec_w = registry[world]
        if not spec_w.engine:
            raise SystemExit(f"world {world!r} has no engine; it can only be ingested from recordings")
        for i in range(episodes):
            s = seed + i
            spec = RolloutSpec(
                world=world,
                engine=spec_w.engine,
                engine_args=dict(spec_w.engine_args),
                seconds=seconds,
                seed=s,
                profile=profile,
                lag_ms=lag_ms,
                drift_ms_per_min=drift_ms_per_min,
                width=size[0],
                height=size[1],
                gt=gt,
                stream=f"{spec_w.engine}:{world}:player{s % max(1, streams)}",
            )
            suffix = ""
            if profile != "player" or lag_ms or drift_ms_per_min:
                suffix = f"-{profile}" + (f"-lag{lag_ms:g}" if lag_ms else "") + (
                    f"-drift{drift_ms_per_min:g}" if drift_ms_per_min else "")
            jobs.append((spec, str(out_root), f"{world}-{s:04d}{suffix}"))
    pool = _pool(workers, len(jobs))
    if pool is None:
        return [_rollout_one(*job) for job in jobs]
    with pool:
        return list(pool.map(_rollout_one, *zip(*jobs)))


def _ingest_rows(metas) -> list[dict[str, Any]]:
    return [
        {"recording_id": m.recording_id, "world": m.world, "duration": m.duration, "stream": m.stream,
         "split_hint": m.split_hint, "session": m.session or m.recording_id}
        for m in (metas if isinstance(metas, list) else [metas])
    ]


def _ingest_one(source: str, path: str, out_root: str, config_dir: str | None) -> list[dict[str, Any]]:
    adapter = resolve("source", source)()
    return _ingest_rows(adapter.convert(Path(path), Path(out_root), WorldRegistry.load(config_dir)))


def run_ingest(
    source: str, input_root: Path, out_root: Path, registry: WorldRegistry, workers: int = 1,
    config_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert every file the adapter finds; ``workers`` > 1 converts in
    parallel (each worker reloads the registry from ``config_dir``)."""
    adapter = resolve("source", source)()
    paths = adapter.discover(Path(input_root))
    pool = _pool(workers, len(paths))
    if pool is None:
        parts = [_ingest_rows(adapter.convert(path, Path(out_root), registry)) for path in paths]
    else:
        cfg = str(config_dir) if config_dir else None
        with pool:
            parts = list(pool.map(_ingest_one, [source] * len(paths), map(str, paths),
                                  [str(out_root)] * len(paths), [cfg] * len(paths)))
    return [row for part in parts for row in part]


# -- curate ---------------------------------------------------------------------

def _curate_one(rec_dir: str, raw_root: str, out_dir: str, qcfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(rec_dir)
    cfg = QualityConfig(**qcfg)
    rec = read_recording(root)
    alignment, events, realigned_ms = check_alignment(rec, root, cfg)
    turns = Canonicalizer().turns(events, rec.observations, rec.meta.duration, rec.meta.mouse_scale)
    report = assess(rec, turns, alignment, realigned_ms, cfg)
    meta = rec.meta
    row = {
        "recording_id": meta.recording_id,
        "world": meta.world,
        "source": meta.source,
        "stream": meta.stream,
        "duration": meta.duration,
        "frame_hz": meta.frame_hz,
        "exact_state": meta.exact_state,
        "split_hint": meta.split_hint,
        "session": meta.session or meta.recording_id,
        "raw_dir": os.path.relpath(root, raw_root),
        "passed": report.passed,
        "turns": len(turns),
        **report.to_json(),
    }
    if report.passed:
        path = Path(out_dir) / "turns" / f"{meta.recording_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for turn in turns:
                fh.write(json.dumps(turn_to_json(turn), ensure_ascii=False) + "\n")
        row["turns_file"] = f"turns/{meta.recording_id}.jsonl"
    return row


def curate(raw_root: Path, out_dir: Path, qcfg: QualityConfig | None = None, workers: int = 1) -> dict[str, Any]:
    qcfg = qcfg or QualityConfig()
    raw_root, out_dir = Path(raw_root).resolve(), Path(out_dir)
    dirs = discover(raw_root)
    if not dirs:
        raise SystemExit(f"no recordings under {raw_root}; run `embodigen rollout` or `embodigen ingest` first")
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "turns"
    if stale.is_dir():
        for p in stale.glob("*.jsonl"):
            p.unlink()
    jobs = [(str(d), str(raw_root), str(out_dir), qcfg.to_json()) for d in dirs]
    pool = _pool(workers, len(jobs))
    if pool is None:
        rows = [_curate_one(*job) for job in jobs]
    else:
        with pool:
            rows = list(pool.map(_curate_one, *zip(*jobs)))
    rows.sort(key=lambda r: r["recording_id"])

    with (out_dir / "recordings.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "rejected.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            if not row["passed"]:
                fh.write(json.dumps({k: row[k] for k in ("recording_id", "world", "reasons", "alignment")},
                                    ensure_ascii=False) + "\n")

    worlds: dict[str, dict[str, Any]] = {}
    for row in rows:
        w = worlds.setdefault(row["world"], {"recordings": 0, "certified": 0, "hours_total": 0.0,
                                             "hours_certified": 0.0, "realigned": 0, "reasons": {},
                                             "sources": [], "turns": 0, "mouse_clipped": 0.0})
        w["recordings"] += 1
        w["hours_total"] += row["duration"] / 3600.0
        if row["source"] not in w["sources"]:
            w["sources"].append(row["source"])
        w["turns"] += row["turns"]
        w["mouse_clipped"] += row["metrics"].get("mouse_clipped_frac", 0.0) * row["turns"]
        if row["passed"]:
            w["certified"] += 1
            w["hours_certified"] += row["duration"] / 3600.0
            w["realigned"] += bool(row["realigned_ms"])
        for reason in row["reasons"]:
            key = reason.split(":")[0]
            w["reasons"][key] = w["reasons"].get(key, 0) + 1
    for w in worlds.values():
        w["pass_rate"] = round(w["certified"] / w["recordings"], 4)
        w["hours_total"] = round(w["hours_total"], 5)
        w["hours_certified"] = round(w["hours_certified"], 5)
        w["mouse_clipped_frac"] = round(w.pop("mouse_clipped") / max(1, w.pop("turns")), 4)
    certified = sum(r["passed"] for r in rows)
    ledger = {
        "stage": "curate",
        "updated_at": _now(),
        "raw_root": str(raw_root),
        "canonicalizer": Canonicalizer.name,
        "alignment_checker": AlignmentChecker.name,
        "quality": qcfg.to_json(),
        "summary": {
            "recordings": len(rows),
            "certified": certified,
            "rejected": len(rows) - certified,
            "yield": round(certified / len(rows), 4),
            "hours_certified": round(sum(r["duration"] for r in rows if r["passed"]) / 3600.0, 5),
            "realigned": sum(1 for r in rows if r["passed"] and r["realigned_ms"]),
            # aligned = measured and in sync; inconclusive/skipped = could not
            # be verified from pixels (e.g. top-down games, no video decoder).
            "alignment": {
                status: sum(1 for r in rows if r["alignment"].get("status") == status)
                for status in sorted({r["alignment"].get("status", "?") for r in rows})
            },
        },
        "worlds": dict(sorted(worlds.items())),
    }
    _write_json(out_dir / "ledger.json", ledger)
    return ledger


# -- build ----------------------------------------------------------------------

def load_study(path: Path) -> StudySpec:
    with Path(path).open("rb") as fh:
        return StudySpec.from_toml(tomllib.load(fh))


class _TurnStore:
    def __init__(self, curated: Path, rows: dict[str, dict[str, Any]]) -> None:
        self.curated, self.rows = curated, rows

    def load(self, recording_id: str) -> list[Turn]:
        path = self.curated / self.rows[recording_id]["turns_file"]
        return [turn_from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def idle_flags(self, recording_id: str) -> list[bool]:
        path = self.curated / self.rows[recording_id]["turns_file"]
        return [json.loads(line)["idle"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(
    curated: Path,
    study: StudySpec,
    registry: WorldRegistry,
    out_dir: Path,
    annotator_name: str | None = None,
) -> dict[str, Any]:
    curated, out_dir = Path(curated), Path(out_dir)
    cur_ledger_path = curated / "ledger.json"
    if not cur_ledger_path.exists():
        raise SystemExit(f"{curated} has no ledger.json; run `embodigen curate` first")
    cur_ledger = json.loads(cur_ledger_path.read_text(encoding="utf-8"))
    raw_root = cur_ledger["raw_root"]
    rows = [r for r in _jsonl(curated / "recordings.jsonl") if r["passed"]]
    by_id = {r["recording_id"]: r for r in rows}
    pass_rates = {w: v["pass_rate"] for w, v in cur_ledger["worlds"].items()}

    wcfg = WindowConfig(**study.windows)
    scfg = SplitConfig(**study.val)
    rate = float(study.reasoning.get("rate", DEFAULT_RATE))
    salt = f"think:{study.reasoning.get('salt', study.seed)}"
    shard_size = int(study.serialize.get("shard_size", 2048))
    annotator = make_annotator(annotator_name or study.reasoning.get("annotator", "mock"), study.reasoning)
    if hasattr(annotator, "check"):
        annotator.check()  # e.g. a cloud annotator without its API key stops here, before any work

    store = _TurnStore(curated, by_id)
    windows: dict[str, list[Window]] = {
        r["recording_id"]: make_windows(r["recording_id"], r["world"], r["stream"], store.idle_flags(r["recording_id"]), wcfg)
        for r in rows
    }
    splits = split(rows, windows, registry, study.anchor, study.ood, pass_rates, scfg, wcfg, seed=study.seed)
    train_rows, capped_out = cap_streams(splits.train, wcfg, salt=f"cap{study.seed}")
    raw_pool = [w for r in train_rows for w in windows[r["recording_id"]]]
    pool = downsample_idle(raw_pool, wcfg, salt=f"idle{study.seed}")
    builder = MixtureBuilder(pool, study.anchor, seed=study.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    # A rebuild replaces every mixture and validation set it writes; shards
    # from an earlier, differently sized build must not linger next to them.
    for stale in [out_dir / "val", *(out_dir / m.name for m in study.mixtures)]:
        if stale.is_dir():
            shutil.rmtree(stale)
    cache = RationaleCache(out_dir / "cache" / f"rationales-{annotator.name}.jsonl")
    media = lambda rid: by_id[rid]["raw_dir"]  # noqa: E731 - tiny adapter
    frame_root = lambda rid: Path(raw_root) / by_id[rid]["raw_dir"]  # noqa: E731
    config_hash = hashlib.sha256(
        json.dumps({"study": asdict(study), "curate": cur_ledger["quality"], "annotator": annotator.name},
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    common = {
        "study": study.name,
        "config_hash": config_hash,
        "media_root": raw_root,
        "template": {
            "user": "<image>",
            "assistant": "<think>\\n[rationale]\\n</think>\\n<|action_start|> dx ; dy ; dz ; k1 ; ... ; k6 <|action_end|>",
            "loss": "assistant turns only; image tokens carry no loss",
            "obs_hz": 5,
            "chunk_ms": 200,
            "substeps": 6,
        },
        "annotator": annotator.name,
    }

    # Annotators that call a service fill the cache concurrently first; the
    # writing passes below then only read it. Budgets are nested, so the
    # largest budget of each mixture covers every turn any run will need.
    prefilled = 0
    if rate > 0 and getattr(annotator, "workers", 1) > 1:
        needed = [w for clips in splits.val.values() for w in clips]
        for mix in study.mixtures:
            needed += builder.select(mix, max(study.budgets_hours), wcfg.budget_unit)["windows"]
        by_rec: dict[str, list[Window]] = {}
        for w in needed:
            by_rec.setdefault(w.recording_id, []).append(w)
        jobs, seen = [], set()
        for rid in sorted(by_rec):
            turns = store.load(rid)
            for w in by_rec[rid]:
                for turn in turns[w.start : w.start + w.turns]:
                    if (rid, turn.index) in seen or not selected(rid, turn.index, rate, salt):
                        continue
                    seen.add((rid, turn.index))
                    context = turns[max(0, turn.index - 10) : turn.index]
                    jobs.append((rid, turn, context, w.world, (frame_root(rid), turn.frame)))
        prefilled = cache.prefill(annotator, jobs, annotator.workers)

    # Validation sets are shared by every mixture and budget.
    val_info = {}
    for level, clips in splits.val.items():
        writer = ShardWriter(out_dir / "val", level, len(clips), max(len(clips), 1), study.seed)
        stats = write_samples(clips, store.load, media, writer, annotator, cache, rate, salt, frame_root)
        shards = writer.close()
        val_info[level] = {
            "clips": len(clips),
            "worlds": sorted({w.world for w in clips}),
            "recordings": len(splits.val_recordings[level]),
            "reasoning": stats,
            "shards": shards,
        }

    grid = []
    for mix in study.mixtures:
        previous: set[str] = set()
        for budget in study.budgets_hours:
            sel = builder.select(mix, budget, wcfg.budget_unit)
            chosen: list[Window] = sel["windows"]
            ids = {w.window_id for w in chosen}
            if not previous <= ids:
                raise AssertionError(f"{mix.name}: budget {budget}h is not a superset of the previous budget")
            previous = ids
            dest = out_dir / mix.name / f"{budget:g}h"
            writer = ShardWriter(dest, "train", len(chosen), shard_size, study.seed)
            stats = write_samples(chosen, store.load, media, writer, annotator, cache, rate, salt, frame_root)
            shards = writer.close()
            (dest / "windows.txt").write_text("".join(f"{i}\n" for i in sorted(ids)), encoding="utf-8")
            div = diversity(chosen)
            manifest = {
                **common,
                "mixture": mix.name,
                "budget_hours": budget,
                "budget_unit": wcfg.budget_unit,
                "footage_hours": round(footage_seconds(chosen) / 3600.0, 5),
                "window_hours": round(len(chosen) * wcfg.seconds / 3600.0, 5),
                "samples": len(chosen),
                "shortfall_hours": round(sel["shortfall_seconds"] / 3600.0, 5),
                "anchor": study.anchor,
                "anchor_share_target": sel["anchor_share_target"],
                "anchor_share": sel["anchor_share"],
                "diversity": div,
                "reasoning": stats,
                "vl_anchor": {
                    "share_of_tokens": study.vl_anchor_share,
                    "note": "fixed vision-language anchor mixed in by the trainer; not part of the gameplay hours",
                },
                "windows_sha256": hashlib.sha256("".join(sorted(ids)).encode()).hexdigest(),
                "shards": shards,
                "created_at": _now(),
            }
            _write_json(dest / "manifest.json", manifest)
            grid.append({
                "mixture": mix.name,
                "budget_hours": budget,
                "footage_hours": manifest["footage_hours"],
                "window_hours": manifest["window_hours"],
                "samples": len(chosen),
                "shortfall_hours": manifest["shortfall_hours"],
                "worlds": div["worlds"],
                "hill_d": div["hill_d"],
                "inverse_simpson": div["inverse_simpson"],
                "largest_share": div["largest_share"],
                "anchor_share": sel["anchor_share"],
                "rationale_density": stats["density"],
                "path": f"{mix.name}/{budget:g}h",
            })
    cache.close()

    densities = [g["rationale_density"] for g in grid if g["samples"]]
    short = [f"{g['mixture']}@{g['budget_hours']:g}h" for g in grid if g["shortfall_hours"] > 0]
    if short:
        splits.notes.append(
            f"pool too small for {', '.join(short)}: those runs are smaller than their budget "
            f"(shares kept; see shortfall_hours)"
        )
    ledger = {
        "stage": "build",
        "updated_at": _now(),
        **common,
        "curated": str(curated.resolve()),
        "pool": {
            "train_recordings": len(splits.train),
            "after_stream_cap": len(train_rows),
            "dropped_by_stream_cap": [r["recording_id"] for r in capped_out],
            "windows_before_idle_downsample": len(raw_pool),
            "windows": len(pool),
            "footage_hours": round(footage_seconds(pool) / 3600.0, 5),
            "window_hours": round(len(pool) * wcfg.seconds / 3600.0, 5),
            "budget_unit": wcfg.budget_unit,
            "available_windows": builder.available(),
            "diversity": diversity(pool),
        },
        "validation": val_info,
        "ood_worlds": splits.ood_worlds,
        "grid": grid,
        "checks": {
            "budgets_nested": True,
            "train_val_disjoint": True,
            "rationale_density_spread": round(max(densities) - min(densities), 5) if densities else 0.0,
            "rationale_cache": {"hits": cache.hits, "misses": cache.misses, "prefilled": prefilled},
        },
        # Settings and token usage of a service-backed annotator; never its key.
        "annotator_config": annotator.describe() if hasattr(annotator, "describe") else {"name": annotator.name},
        "notes": splits.notes,
    }
    _write_json(out_dir / "ledger.json", ledger)
    return ledger
