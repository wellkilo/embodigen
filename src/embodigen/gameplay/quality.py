"""Quality gate for recordings.

Two kinds of checks, and a recording has to clear both before any of its
windows can reach a mixture:

* **Capture hygiene** -- the rejection rules GameScaling applies to its
  recording pools: input that went to another application, cursor captures
  that warp to the screen centre, too little input, too many idle windows.
* **Action-frame alignment** -- the gameplay counterpart of the robot
  pipeline's fidelity check. It reads only the pixels and the event log:
  the view's horizontal shift between consecutive frames is correlated with
  the logged mouse ``dx`` at a range of lags (both series differenced and
  ranked, so the peak is sharp and outliers cannot drag it). A capture whose input log runs
  ahead of or behind its video is caught here, and if the lag is clear it is
  corrected and re-checked (recorded in the report), the same way a failed
  render is retried with stronger conditioning.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embodigen.gameplay.actions import MOUSE_LIMIT
from embodigen.gameplay.types import (
    BUTTON,
    FOCUS,
    KEY,
    MOVE,
    WHEEL,
    InputEvent,
    Observation,
    QualityReport,
    Recording,
    Turn,
)
from embodigen.gameplay.video import FrameReader, VideoError, png_profile, split_ref, video_reader


@dataclass
class QualityConfig:
    max_non_game_frac: float = 0.05
    max_recenter_frac: float = 0.5
    min_events_per_min: float = 60.0
    min_key_click_per_min: float = 3.0
    max_idle_window_frac: float = 0.6
    idle_turn_frac: float = 0.5  # a window is idle-dominated at this share of idle turns
    window_turns: int = 20
    window_stride: int = 10
    min_duration_s: float = 4.0
    alignment: bool = True
    realign: bool = True
    max_lag_ms: float = 600.0
    lag_tolerance_frames: float = 1.0
    # Floor in ms: one 33 ms key slot is the finest timing an action packet
    # records, so a smaller lag changes no label. Matters for 60 fps capture.
    lag_tolerance_ms: float = 34.0
    min_corr: float = 0.2
    min_motion_std: float = 2.0
    align_max_frames: int = 450  # frames per measured segment
    align_segments: int = 3  # start / middle / end, when the recording is long enough
    # Segments with camera motion but no correlation next to a measured one
    # are approached in this many steps, each searching around the last lag:
    # that follows a drifting clock past the +-max_lag_ms window.
    align_probes: int = 4
    profile_columns: int = 80
    video_backend: str = "pyav"  # decodes ``<file>#t=`` frames; "" skips video-only recordings
    # Rows used for the view profile: skips the floor (it streams past when
    # walking) and the bottom HUD most games draw.
    profile_band: tuple[float, float] = (0.1, 0.6)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def idle_window_fraction(turns: list[Turn], cfg: QualityConfig) -> float:
    n = cfg.window_turns
    starts = range(0, len(turns) - n + 1, cfg.window_stride)
    windows = [sum(t.action.idle for t in turns[s : s + n]) / n for s in starts]
    if not windows:
        return 1.0
    return sum(1 for f in windows if f >= cfg.idle_turn_frac) / len(windows)


def capture_metrics(rec: Recording, turns: list[Turn], cfg: QualityConfig) -> dict[str, float]:
    meta = rec.meta
    minutes = max(meta.duration, 1e-6) / 60.0
    focus_changes = sorted((e.t, e.code) for e in rec.events if e.kind == FOCUS)
    focus_t = [ft for ft, _ in focus_changes]

    def in_game(t: float) -> bool:
        i = bisect.bisect_right(focus_t, t) - 1
        return (focus_changes[i][1] if i >= 0 else meta.app) == meta.app

    inputs = [
        e for e in rec.events
        if e.kind in (MOVE, WHEEL) or (e.kind in (KEY, BUTTON) and e.down)
    ]
    non_game = sum(1 for e in inputs if not in_game(e.t))
    presses = [e for e in inputs if e.kind in (KEY, BUTTON) and in_game(e.t)]
    abs_moves = [e for e in rec.events if e.kind == MOVE and e.x is not None and e.y is not None]
    cx, cy = meta.width / 2.0, meta.height / 2.0
    recentred = sum(1 for e in abs_moves if abs(e.x - cx) <= 2 and abs(e.y - cy) <= 2)  # type: ignore[operator]
    return {
        "duration_s": round(meta.duration, 3),
        "input_events": len(inputs),
        "events_per_min": round(len(inputs) / minutes, 2),
        "key_click_per_min": round(len(presses) / minutes, 2),
        "non_game_frac": round(non_game / len(inputs), 4) if inputs else 0.0,
        "recenter_frac": round(recentred / len(abs_moves), 4) if abs_moves else 0.0,
        "idle_window_frac": round(idle_window_fraction(turns, cfg), 4),
        "idle_turn_frac": round(sum(t.action.idle for t in turns) / len(turns), 4) if turns else 1.0,
        "turns": len(turns),
        "truncated_keys": sum(t.truncated_keys for t in turns),
        # Packets whose mouse motion hit the +-1000 clip: a hint to tune the
        # source's mouse_scale, not a rejection reason.
        "mouse_clipped_frac": round(
            sum(1 for t in turns if max(abs(t.action.dx), abs(t.action.dy)) >= MOUSE_LIMIT) / len(turns), 4
        ) if turns else 0.0,
    }


def capture_reasons(m: dict[str, float], cfg: QualityConfig) -> list[str]:
    reasons = []
    if m["duration_s"] < cfg.min_duration_s:
        reasons.append(f"too_short: {m['duration_s']:.1f}s < {cfg.min_duration_s:g}s")
    if m["non_game_frac"] >= cfg.max_non_game_frac:
        reasons.append(f"non_game_input: {m['non_game_frac']:.1%} of input went to other apps (limit {cfg.max_non_game_frac:.0%})")
    if m["recenter_frac"] >= cfg.max_recenter_frac:
        reasons.append(f"cursor_recentering: {m['recenter_frac']:.0%} of mouse events (limit {cfg.max_recenter_frac:.0%})")
    if m["events_per_min"] < cfg.min_events_per_min:
        reasons.append(f"sparse_input: {m['events_per_min']:.0f} events/min < {cfg.min_events_per_min:g}")
    if m["key_click_per_min"] < cfg.min_key_click_per_min:
        reasons.append(f"sparse_keys: {m['key_click_per_min']:.1f} keys+clicks/min < {cfg.min_key_click_per_min:g}")
    if m["idle_window_frac"] > cfg.max_idle_window_frac:
        reasons.append(f"idle_windows: {m['idle_window_frac']:.0%} of windows idle (limit {cfg.max_idle_window_frac:.0%})")
    return reasons


# -- alignment ----------------------------------------------------------------

def _shift(a: list[float], b: list[float], max_shift: int) -> float:
    """Horizontal shift of ``b`` against ``a``, in profile columns, refined to
    sub-column precision with a parabola through the best three costs."""
    n = len(a)
    costs: dict[int, float] = {}
    for s in range(-max_shift, max_shift + 1):
        lo, hi = max(0, -s), min(n, n - s)
        if hi - lo < n // 2:
            continue
        costs[s] = sum(abs(a[x] - b[x + s]) for x in range(lo, hi)) / (hi - lo)
    best = min(costs, key=costs.__getitem__)
    if best - 1 in costs and best + 1 in costs:
        c0, c1, c2 = costs[best - 1], costs[best], costs[best + 1]
        curve = c0 - 2 * c1 + c2
        if curve > 0:
            return best + 0.5 * (c0 - c2) / curve
    return float(best)


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so the correlation below is Spearman's: robust to the
    occasional wild shift estimate when the view is mostly one flat wall."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


class AlignmentChecker:
    """Measures the input log's lag against the video in a few segments.

    One constant lag is a capture offset and can be corrected. Lags that
    disagree between the start, middle and end of a recording are clock
    drift: no single shift fixes that, so it is reported as ``drift``.
    """

    name = "flow-mouse-xcorr-v2"

    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg
        self._profiles: dict[str, list[float]] = {}
        self._video: FrameReader | None = None
        self._video_note = ""

    def _reader(self) -> FrameReader | None:
        if self._video is None and not self._video_note:
            self._video, self._video_note = video_reader(self.cfg.video_backend)
        return self._video

    def frames(self, rec: Recording, root: Path) -> tuple[list[Observation], str]:
        """Observations whose pixels can be read, or ([], why not)."""
        pngs = [o for o in rec.observations if o.frame.endswith(".png")]
        if pngs:
            return (pngs, "") if (root / pngs[0].frame).exists() else ([], "frames missing on disk")
        videos = [o for o in rec.observations if split_ref(o.frame)[1] is not None]
        if not videos:
            return [], "no decoded frames"
        if not (root / split_ref(videos[0].frame)[0]).exists():
            return [], "video missing: frames are references only"
        if self._reader() is None:
            return [], self._video_note
        return videos, ""

    def tolerance_ms(self, frame_ms: float) -> float:
        return max(self.cfg.lag_tolerance_frames * frame_ms, self.cfg.lag_tolerance_ms)

    def starts(self, total: int, n: int) -> list[int]:
        count = max(1, min(self.cfg.align_segments, total // n))
        if count == 1:
            return [(total - n) // 2]  # the middle: start-up frames are often static
        return [round(i * (total - n) / (count - 1)) for i in range(count)]

    def view_shifts(self, frames: list[Observation], root: Path) -> list[float]:
        band = tuple(self.cfg.profile_band)
        todo = [o for o in frames if str(root / o.frame) not in self._profiles]
        if todo and not todo[0].frame.endswith(".png"):
            reader = self._reader()
            assert reader is not None
            for o, prof in zip(todo, reader.profiles(root, todo, self.cfg.profile_columns, band)):
                self._profiles[str(root / o.frame)] = prof
        for o in todo:
            key = str(root / o.frame)
            if key not in self._profiles:
                self._profiles[key] = png_profile(root / o.frame, self.cfg.profile_columns, band)
        profiles = [self._profiles[str(root / o.frame)] for o in frames]
        span = len(profiles[0]) // 3
        return [_shift(profiles[i], profiles[i + 1], span) for i in range(len(profiles) - 1)]

    def measure_segment(self, times: list[float], shifts: list[float], moves: list[tuple[float, float]],
                        center_ms: float = 0.0) -> dict[str, Any]:
        dxs = []
        j = 0
        eps = 1e-6  # timestamps are logged to the microsecond
        for i in range(len(times) - 1):
            while j < len(moves) and moves[j][0] < times[i] - eps:
                j += 1
            k, total = j, 0.0
            while k < len(moves) and moves[k][0] < times[i + 1] - eps:
                total += moves[k][1]
                k += 1
            dxs.append(total)
        frame_dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        if _std(dxs) < self.cfg.min_motion_std or _std(shifts) == 0.0:
            return {"status": "inconclusive", "reason": "not enough camera motion", "frames": len(times),
                    "t0": round(times[0], 3)}
        max_lag = max(1, int(round(self.cfg.max_lag_ms / 1000.0 / frame_dt)))
        center = int(round(-center_ms / 1000.0 / frame_dt))
        # Pre-whiten by differencing: mouse motion is smooth, so raw series
        # correlate over a broad plateau of lags and the peak can sit a few
        # frames off. Changes in motion line up only at the true lag.
        dxs = [dxs[i + 1] - dxs[i] for i in range(len(dxs) - 1)]
        shifts = [shifts[i + 1] - shifts[i] for i in range(len(shifts) - 1)]
        dxs, shifts = _ranks(dxs), _ranks(shifts)
        best_lag, best = 0, 0.0
        for lag in range(center - max_lag, center + max_lag + 1):
            if abs(lag) > len(dxs) // 2:  # keep at least half the segment overlapping
                continue
            if lag >= 0:
                a, b = dxs[: len(dxs) - lag], shifts[lag:]
            else:
                a, b = dxs[-lag:], shifts[: len(shifts) + lag]
            c = abs(_corr(a, b))
            if c > best:
                best_lag, best = lag, c
        if best < self.cfg.min_corr:
            status = "inconclusive"
        elif abs(best_lag) * frame_dt * 1000.0 <= self.tolerance_ms(frame_dt * 1000.0) + 1e-6:
            status = "aligned"
        else:
            status = "misaligned"
        return {
            "status": status,
            "lag_ms": round(-best_lag * frame_dt * 1000.0, 1),  # >0: the input log runs late
            "corr": round(best, 3),
            "frames": len(times),
            "frame_ms": round(frame_dt * 1000.0, 2),
            "t0": round(times[0], 3),
        }

    def measure(self, rec: Recording, root: Path, events: list[InputEvent]) -> dict[str, Any]:
        try:
            return self._measure(rec, root, events)
        except VideoError as exc:
            return {"status": "skipped", "reason": f"video unreadable: {exc}", "checker": self.name}

    def _measure(self, rec: Recording, root: Path, events: list[InputEvent]) -> dict[str, Any]:
        frames, why = self.frames(rec, root)
        if len(frames) < 20:
            return {"status": "skipped", "reason": why or "too few frames", "checker": self.name}
        n = min(len(frames), self.cfg.align_max_frames)
        moves = sorted((e.t, e.dx) for e in events if e.kind == MOVE)

        def run(start: int, center_ms: float = 0.0) -> dict[str, Any]:
            seg = frames[start : start + n]
            part = self.measure_segment([o.t for o in seg], self.view_shifts(seg, root), moves, center_ms)
            return {**part, "start": start}

        parts = [run(s) for s in self.starts(len(frames), n)]
        probes = self._track(parts, run)
        conclusive = [p for p in parts + probes if p["status"] in ("aligned", "misaligned")]
        out: dict[str, Any] = {"checker": self.name}
        if len(parts) > 1:
            out["segments"] = [_public(p) for p in parts]
        if probes:
            out["probes"] = [_public(p) for p in probes]
        if not conclusive:
            return {**out, "status": "inconclusive", "reason": parts[0].get("reason", "weak correlation")}
        lags = sorted(p["lag_ms"] for p in conclusive)
        frame_ms = conclusive[0]["frame_ms"]
        out.update(
            lag_ms=lags[len(lags) // 2],
            corr=round(min(p["corr"] for p in conclusive), 3),
            frames=sum(p["frames"] for p in parts),
            frame_ms=frame_ms,
        )
        # A constant offset is only established when every segment agrees.
        out["complete"] = all(p["status"] in ("aligned", "misaligned") for p in parts)
        if lags[-1] - lags[0] > 2 * self.tolerance_ms(frame_ms) + frame_ms:
            out["status"] = "drift"
        elif all(p["status"] == "aligned" for p in conclusive):
            out["status"] = "aligned"
        else:
            out["status"] = "misaligned"
        return out

    def _track(self, parts: list[dict[str, Any]], run) -> list[dict[str, Any]]:
        """Walk from a measured segment towards each moving-but-uncorrelated
        one, re-centring the lag search on every step. A segment reached this
        way replaces the weak result; the steps are returned as probes."""
        probes: list[dict[str, Any]] = []
        measured = [i for i, p in enumerate(parts) if p["status"] in ("aligned", "misaligned")]
        for i, part in enumerate(parts):
            if part["status"] != "inconclusive" or "corr" not in part or not measured:
                continue
            j = min(measured, key=lambda m: abs(m - i))
            a, b = parts[j]["start"], part["start"]
            center = parts[j]["lag_ms"]
            steps = self.cfg.align_probes
            for k in range(1, steps + 2):
                probe = run(round(a + (b - a) * k / (steps + 1)), center)
                if probe["status"] in ("aligned", "misaligned"):
                    center = probe["lag_ms"]
                if k == steps + 1:
                    if probe["status"] != "inconclusive":
                        parts[i] = {**probe, "tracked": True}
                else:
                    probes.append(probe)
        return probes


def _public(part: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in part.items() if k != "start"}


def shift_events(events: list[InputEvent], seconds: float, duration: float) -> list[InputEvent]:
    out = []
    for e in events:
        t = round(e.t - seconds, 6)  # stay on the microsecond grid the logs use
        if 0.0 <= t < duration:
            out.append(InputEvent(t=t, kind=e.kind, code=e.code, down=e.down, dx=e.dx, dy=e.dy, dz=e.dz, x=e.x, y=e.y))
    return out


def check_alignment(rec: Recording, root: Path, cfg: QualityConfig) -> tuple[dict[str, Any], list[InputEvent], float]:
    """Measure alignment; realign once if the lag is clear. Returns (report, events, correction_ms)."""
    if not cfg.alignment:
        return {"status": "disabled"}, rec.events, 0.0
    checker = AlignmentChecker(cfg)
    first = checker.measure(rec, root, rec.events)
    if first["status"] != "misaligned" or not cfg.realign:
        return first, rec.events, 0.0
    if not first.get("complete", True):
        # Some segments could not be measured, so a drifting clock cannot be
        # ruled out: shifting the whole log by one segment's lag could make
        # the rest worse. Reject rather than guess.
        first["note"] = "lag measured in only part of the recording; not corrected"
        return first, rec.events, 0.0
    lag_ms = first["lag_ms"]
    corrected = shift_events(rec.events, lag_ms / 1000.0, rec.meta.duration)
    second = checker.measure(rec, root, corrected)
    second["before"] = {"lag_ms": first["lag_ms"], "corr": first["corr"]}
    if second["status"] == "aligned":
        return second, corrected, lag_ms
    second["status"] = "misaligned"
    return second, rec.events, 0.0


def assess(
    rec: Recording,
    turns: list[Turn],
    alignment: dict[str, Any],
    realigned_ms: float,
    cfg: QualityConfig,
) -> QualityReport:
    metrics = capture_metrics(rec, turns, cfg)
    reasons = capture_reasons(metrics, cfg)
    if alignment.get("status") == "drift":
        lags = [p["lag_ms"] for p in alignment.get("segments", []) if "lag_ms" in p]
        reasons.append(f"clock_drift: input/video lag moves across the recording ({lags} ms)")
    if alignment.get("status") == "misaligned":
        lag = alignment.get("before", alignment).get("lag_ms", "?")
        if not alignment.get("complete", True):
            reasons.append(f"desync: input log off by {lag} ms where measurable; rest unverified, not corrected")
        else:
            reasons.append(f"desync: input log off by {lag} ms and not correctable")
    return QualityReport(
        passed=not reasons,
        reasons=reasons,
        metrics=metrics,
        alignment=alignment,
        realigned_ms=realigned_ms,
    )
