#!/usr/bin/env python3
"""How well does the alignment gate catch an input log that runs early or late?

For every raw recording that has readable video, shift its input events by a
known lag and run the real gate (``check_alignment``: measure, realign once,
re-measure). The recording's own lag is measured first (lag 0), so the
injected lag is judged against ``baseline + injected``. Outcomes:

    recovered   realigned, and the correction is within the gate's tolerance of the true lag
    rejected    marked misaligned / drift: the recording would not be certified
    missed      passed as aligned without being corrected, or came back inconclusive
    wrong       realigned by an amount that is off by more than the tolerance

A lag within the gate's tolerance (34 ms) is expected to pass untouched: it changes no label.
Needs ``.[video]`` (PyAV) for recordings whose frames are inside a video.

    python3 scripts/alignment_sweep.py --raw data/samples-raw --out data/alignment-sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embodigen.gameplay.quality import AlignmentChecker, QualityConfig, check_alignment, shift_events
from embodigen.gameplay.recording import read_recording

LAGS_MS = (-600, -300, -150, -100, -50, 0, 50, 100, 150, 300, 600)


def classify(first: dict, report: dict, corrected_ms: float, injected: float, baseline: float, tol: float) -> str:
    true_lag = injected + baseline
    status = report.get("status")
    if abs(true_lag) <= tol:
        return "ok" if status == "aligned" and not corrected_ms else "ok-corrected" if status == "aligned" else "false-alarm"
    if status in ("misaligned", "drift"):
        return "rejected"
    if status == "aligned" and corrected_ms:
        return "recovered" if abs(corrected_ms - true_lag) <= tol else "wrong"
    return "missed"


def sweep(raw: Path, lags: tuple[int, ...] = LAGS_MS, cfg: QualityConfig | None = None) -> list[dict]:
    cfg = cfg or QualityConfig()
    rows = []
    for rec_dir in sorted(p.parent for p in raw.rglob("recording.json")):
        rec = read_recording(rec_dir)
        checker = AlignmentChecker(cfg)  # one checker per recording: frames are decoded once
        base, _, _ = check_alignment(rec, rec_dir, cfg, checker)
        row = {"recording_id": rec.meta.recording_id, "source": rec.meta.source, "world": rec.meta.world,
               "baseline": {k: base.get(k) for k in ("status", "lag_ms", "corr")}, "lags": []}
        if base.get("status") not in ("aligned", "misaligned"):
            row["skipped"] = base.get("reason") or base.get("status")
            rows.append(row)
            print(f"{rec.meta.recording_id}: skipped ({row['skipped']})", file=sys.stderr)
            continue
        baseline = base.get("before", base).get("lag_ms", 0.0) or 0.0
        tol = checker.tolerance_ms(base.get("frame_ms", 33.3))
        for lag in lags:
            events = shift_events(rec.events, -lag / 1000.0, rec.meta.duration)
            shifted = type(rec)(meta=rec.meta, events=events, observations=rec.observations)
            report, _, corrected = check_alignment(shifted, rec_dir, cfg, checker)
            row["lags"].append({
                "injected_ms": lag, "outcome": classify(base, report, corrected, lag, baseline, tol),
                "status": report.get("status"), "corrected_ms": corrected,
                "measured_ms": (report.get("before") or report).get("lag_ms"),
            })
        row["baseline_ms"], row["tolerance_ms"] = baseline, tol
        rows.append(row)
        print(f"{rec.meta.recording_id}: done", file=sys.stderr)
    return rows


def table(rows: list[dict]) -> str:
    measured = [r for r in rows if r["lags"]]
    lags = sorted({x["injected_ms"] for r in measured for x in r["lags"]})
    out = [f"{len(measured)} recordings swept, {len(rows) - len(measured)} not measurable (no usable camera motion or video)", ""]
    out.append("injected ms  " + "  ".join(f"{lag:>5d}" for lag in lags))
    for r in measured:
        cells = {x["injected_ms"]: x["outcome"] for x in r["lags"]}
        out.append(f"{r['recording_id'][:11]:<11}  " + "  ".join(f"{cells[lag][:5]:>5}" for lag in lags))
    big = [x for r in measured for x in r["lags"] if abs(x["injected_ms"] + r["baseline_ms"]) > r["tolerance_ms"]]
    small = [x for r in measured for x in r["lags"] if abs(x["injected_ms"] + r["baseline_ms"]) <= r["tolerance_ms"]]
    count = lambda xs, *names: sum(x["outcome"] in names for x in xs)  # noqa: E731
    out += ["",
            f"lag beyond tolerance ({len(big)} trials): recovered {count(big, 'recovered')}, rejected {count(big, 'rejected')}, "
            f"wrong {count(big, 'wrong')}, missed {count(big, 'missed')}",
            f"lag within tolerance ({len(small)} trials): untouched {count(small, 'ok')}, corrected anyway {count(small, 'ok-corrected')}, "
            f"false alarm {count(small, 'false-alarm')}"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default="data/samples-raw")
    ap.add_argument("--out", default="")
    ap.add_argument("--lags", default=",".join(map(str, LAGS_MS)), help="comma-separated injected lags in ms")
    args = ap.parse_args(argv)
    rows = sweep(Path(args.raw), tuple(int(v) for v in args.lags.split(",")))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
