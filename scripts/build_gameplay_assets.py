#!/usr/bin/env python3
"""Export the virtual-world numbers the showcase page shows, from real runs.

Reads the ledgers that `make worlds` and `make samples VIDEO=--video` leave
under data/, and writes docs/assets/gameplay.js (``window.GAMEPLAY``). With
``--inject`` it also re-runs the alignment check on real recordings whose
input logs are shifted or stretched, the experiment the README table quotes
(needs the sample videos and the [video] extra).

    python3 scripts/build_gameplay_assets.py --inject
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodigen.gameplay.quality import QualityConfig, check_alignment  # noqa: E402
from embodigen.gameplay.recording import read_recording  # noqa: E402

INJECT = {
    "vpt": "vpt-Player129-f153ac423f61-20210617-172319",
    "p2p": "p2p-01993bb8-12e7-78a1-b661-0c0c3a258e21",
    "d2e": "d2e-minecraft-20250827_052328_366c9184_split_00",
}
CASES = [("+200 ms", 200.0, 0.0), ("−150 ms", -150.0, 0.0), ("+400 ms", 400.0, 0.0), ("600 ms/min", 0.0, 600.0)]


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `make worlds` and `make samples VIDEO=--video` first")
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inject(raw: Path) -> list[dict]:
    rows = []
    for source, rid in INJECT.items():
        src = raw / rid
        natural, _, fixed0 = check_alignment(read_recording(src), src, QualityConfig())
        row = {"source": source, "recording": rid, "natural": {
            "status": natural["status"], "realigned_ms": fixed0,
            "segments": [p.get("lag_ms") for p in natural.get("segments", [natural])]}, "cases": []}
        for label, lag_ms, drift in CASES:
            with tempfile.TemporaryDirectory() as tmp:
                dst = Path(tmp) / rid
                dst.mkdir()
                for f in src.iterdir():
                    if f.name == "events.jsonl":
                        continue
                    if f.is_symlink():
                        os.symlink(os.readlink(f), dst / f.name)
                    elif f.is_file():
                        shutil.copy(f, dst / f.name)
                with (src / "events.jsonl").open() as fi, (dst / "events.jsonl").open("w") as fo:
                    for line in fi:
                        e = json.loads(line)
                        e["t"] = round(e["t"] + lag_ms / 1000 + drift / 1000 * e["t"] / 60, 6)
                        fo.write(json.dumps(e) + "\n")
                a, _, fixed = check_alignment(read_recording(dst), dst, QualityConfig())
            row["cases"].append({"case": label, "status": a["status"], "realigned_ms": fixed,
                                 "accepted": a["status"] in ("aligned", "inconclusive")})
        rows.append(row)
        print(f"  {source}: {[(c['case'], c['status'], c['realigned_ms']) for c in row['cases']]}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "gameplay.js"))
    ap.add_argument("--inject", action="store_true", help="re-run the injected-lag experiment (needs videos + [video])")
    args = ap.parse_args()
    data = Path(args.data)

    ref_cur = load(data / "curated" / "ledger.json")
    ref_rej = jsonl(data / "curated" / "rejected.jsonl")
    ref_build = load(data / "mixes" / "reference" / "ledger.json")
    cur = load(data / "samples-curated" / "ledger.json")
    recs = jsonl(data / "samples-curated" / "recordings.jsonl")
    build = load(data / "samples-mixes" / "public-samples" / "ledger.json")

    top = max(g["budget_hours"] for g in build["grid"])
    grid = [
        {k: g[k] for k in ("mixture", "budget_hours", "footage_hours", "samples", "worlds", "hill_d",
                           "anchor_share", "largest_share", "rationale_density", "shortfall_hours")}
        for g in build["grid"]
    ]
    out = {
        "reference": {
            "recordings": ref_cur["summary"]["recordings"],
            "certified": ref_cur["summary"]["certified"],
            "realigned": ref_cur["summary"]["realigned"],
            "rejected": [{"recording": r["recording_id"], "reason": r["reasons"][0].split(":")[0]} for r in ref_rej],
            "checks": {k: ref_build["checks"][k] for k in ("budgets_nested", "train_val_disjoint")},
        },
        "samples": {
            "summary": cur["summary"],
            "worlds": [
                {"world": w, "sources": v.get("sources", []), "minutes": round(v["hours_certified"] * 60, 1),
                 "recordings": v["recordings"], "clip": v.get("mouse_clipped_frac", 0.0)}
                for w, v in cur["worlds"].items()
            ],
            "recordings": [
                {"recording": r["recording_id"], "source": r["source"], "world": r["world"],
                 "minutes": round(r["duration"] / 60, 1), "alignment": r["alignment"].get("status"),
                 "realigned_ms": r["realigned_ms"]}
                for r in recs
            ],
            "pool": {"windows": build["pool"]["windows"], "minutes": round(build["pool"]["footage_hours"] * 60, 1),
                     "hill_d": build["pool"]["diversity"]["hill_d"], "worlds": build["pool"]["diversity"]["worlds"]},
            "grid": grid,
            "top_budget": top,
            "checks": {k: build["checks"][k] for k in ("budgets_nested", "train_val_disjoint",
                                                        "rationale_density_spread")},
            "annotator": build["annotator"],
        },
    }
    # One real training row: a real game and real recorded actions; the
    # rationale text is whatever annotator the build used (mock by default).
    sample = None
    mix_dir = data / "samples-mixes" / "public-samples" / "S3" / f"{top:g}h"
    for shard in sorted(mix_dir.glob("train-*.jsonl")):
        for row in jsonl(shard):
            assistant = [m["content"] for m in row["messages"][1::2]]
            k = next((i for i, c in enumerate(assistant) if c.count("\n") > 2), None)
            if k is not None and row["world"] != "minecraft" and k + 2 < len(assistant):
                sample = {"world": row["world"], "recording": row["recording"], "start_turn": row["start_turn"] + k,
                          "image": row["images"][k], "turns": assistant[k : k + 3]}
                break
        if sample:
            break
    out["samples"]["example"] = sample

    previous = {}
    target = Path(args.out)
    if target.exists():
        text = target.read_text(encoding="utf-8")
        previous = json.loads(text[text.index("{") : text.rindex("}") + 1]).get("injection", [])
    out["injection"] = inject(data / "samples-raw") if args.inject else previous
    target.write_text(
        "// Generated by scripts/build_gameplay_assets.py from the ledgers of real runs. Do not edit.\n"
        f"window.GAMEPLAY = {json.dumps(out, ensure_ascii=False, indent=1)};\n",
        encoding="utf-8",
    )
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
