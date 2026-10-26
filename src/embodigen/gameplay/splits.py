"""Recording-level splits and the three validation levels.

GameScaling evaluates loss at three distances from the training data:

* ``anchor``    -- the anchor world (Minecraft in the paper; the VPT test split);
* ``in_corpus`` -- held-out recordings of worlds that are in the training pool,
  game-balanced, plus a size-matched anchor subset;
* ``ood``       -- worlds absent from every mixture, de-duplicated against the
  pool by title, alias, franchise and emulated title.

Everything here splits by **recording**, never by window: overlapping windows
of one recording can never sit on both sides. Recordings cut from one play
session (a source that ships a long session as ``_split_00``, ``_split_01``,
...) carry the same ``session`` and move as one unit. :func:`assert_disjoint`
makes both a hard failure rather than a convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.types import Window
from embodigen.gameplay.windows import WindowConfig, h01, is_idle


@dataclass
class SplitConfig:
    clips_per_world: int = 160
    in_corpus_worlds: int = 30
    min_title_pass_rate: float = 0.6


@dataclass
class Splits:
    train: list[dict[str, Any]]  # recording rows
    val: dict[str, list[Window]]  # level -> clips
    val_recordings: dict[str, list[str]]
    ood_worlds: list[str]
    notes: list[str] = field(default_factory=list)
    sessions: dict[str, str] = field(default_factory=dict)  # recording -> session


def check_ood(registry: WorldRegistry, ood: list[str], pool: list[str]) -> None:
    """Refuse an OOD world that is the same game as anything in the pool."""
    for o in ood:
        for p in pool:
            reason = registry.get(o).overlap(registry.get(p))
            if reason:
                raise SystemExit(f"OOD world {o!r} overlaps training world {p!r}: {reason}")


def _clips(windows: list[Window], n: int, wcfg: WindowConfig, salt: str) -> list[Window]:
    """``n`` non-idle clips, chosen by hash so the choice is stable."""
    live = [w for w in windows if not is_idle(w, wcfg)]
    return sorted(live, key=lambda w: h01(salt, w.window_id))[:n]


def session_of(row: dict[str, Any]) -> str:
    return row.get("session") or row["recording_id"]


def _hold_out(
    rows: list[dict[str, Any]], windows: dict[str, list[Window]], need: int, wcfg: WindowConfig, salt: str
) -> list[dict[str, Any]]:
    """Whole sessions (hash order) until they hold ``need`` non-idle clips; always leaves one for training."""
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sessions.setdefault(session_of(row), []).append(row)
    held, have = [], 0
    for key in sorted(sessions, key=lambda k: h01(salt, k))[: max(0, len(sessions) - 1)]:
        if have >= need:
            break
        for row in sessions[key]:
            held.append(row)
            have += sum(1 for w in windows[row["recording_id"]] if not is_idle(w, wcfg))
    return held


def split(
    rows: list[dict[str, Any]],
    windows: dict[str, list[Window]],
    registry: WorldRegistry,
    anchor: str,
    ood: list[str],
    pass_rates: dict[str, float],
    cfg: SplitConfig,
    wcfg: WindowConfig,
    seed: int = 0,
) -> Splits:
    notes: list[str] = []
    by_world: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_world.setdefault(row["world"], []).append(row)

    pool_worlds = sorted(w for w in by_world if w not in ood)
    check_ood(registry, ood, pool_worlds)

    val: dict[str, list[Window]] = {"anchor": [], "in_corpus": [], "ood": []}
    val_recs: dict[str, list[str]] = {"anchor": [], "in_corpus": [], "ood": []}
    held: set[str] = set()

    # OOD: whole worlds, only titles whose recordings mostly pass the gate.
    ood_kept = []
    for world in ood:
        rate = pass_rates.get(world, 0.0)
        if world not in by_world:
            notes.append(f"ood world {world} has no certified recordings")
            continue
        if rate < cfg.min_title_pass_rate:
            notes.append(f"ood world {world} dropped: pass rate {rate:.0%} < {cfg.min_title_pass_rate:.0%}")
            continue
        ood_kept.append(world)
        wins = [w for r in by_world[world] for w in windows[r["recording_id"]]]
        val["ood"].extend(_clips(wins, cfg.clips_per_world, wcfg, f"ood{seed}"))
        val_recs["ood"].extend(r["recording_id"] for r in by_world[world])
        held.update(r["recording_id"] for r in by_world[world])

    # Anchor: the source's own split if it has one, otherwise hash hold-out.
    if anchor in by_world:
        tagged_sessions = {session_of(r) for r in by_world[anchor] if r.get("split_hint") == "val"}
        tagged = [r for r in by_world[anchor] if session_of(r) in tagged_sessions]
        anchor_rows = tagged or _hold_out(by_world[anchor], windows, cfg.clips_per_world, wcfg, f"anchor{seed}")
        anchor_wins = [w for r in anchor_rows for w in windows[r["recording_id"]]]
        val["anchor"] = _clips(anchor_wins, cfg.clips_per_world, wcfg, f"anchor{seed}")
        val_recs["anchor"] = [r["recording_id"] for r in anchor_rows]
        held.update(val_recs["anchor"])
        if not anchor_rows:
            notes.append(f"anchor world {anchor} has a single recording: no anchor validation")
    elif anchor:
        notes.append(f"anchor world {anchor} has no certified recordings")

    # In-corpus: game-balanced held-out recordings of pool worlds, plus a
    # size-matched anchor subset drawn from the anchor hold-out.
    candidates = sorted((w for w in pool_worlds if w != anchor), key=lambda w: h01(f"incorpus{seed}", w))
    chosen = 0
    for world in candidates:
        if chosen >= cfg.in_corpus_worlds:
            break
        rows_w = [r for r in by_world[world] if r["recording_id"] not in held]
        hold = _hold_out(rows_w, windows, cfg.clips_per_world, wcfg, f"incorpus{seed}")
        if not hold:
            notes.append(f"in-corpus: {world} has one recording, kept for training")
            continue
        wins = [w for r in hold for w in windows[r["recording_id"]]]
        val["in_corpus"].extend(_clips(wins, cfg.clips_per_world, wcfg, f"incorpus{seed}"))
        val_recs["in_corpus"].extend(r["recording_id"] for r in hold)
        held.update(r["recording_id"] for r in hold)
        chosen += 1
    if val["anchor"]:
        val["in_corpus"].extend(val["anchor"][: cfg.clips_per_world])

    held_sessions = {session_of(r) for r in rows if r["recording_id"] in held}
    train = [r for r in rows if r["world"] in pool_worlds and session_of(r) not in held_sessions]
    out = Splits(train=train, val=val, val_recordings=val_recs, ood_worlds=ood_kept, notes=notes)
    out.sessions = {r["recording_id"]: session_of(r) for r in rows}
    assert_disjoint(out)
    return out


def assert_disjoint(splits: Splits) -> None:
    train = {r["recording_id"] for r in splits.train}
    train_sessions = {session_of(r) for r in splits.train}
    for level, clips in splits.val.items():
        leaked = train & {w.recording_id for w in clips}
        if leaked:
            raise AssertionError(f"{level} validation shares recordings with train: {sorted(leaked)[:5]}")
        leaked = train_sessions & {splits.sessions.get(w.recording_id, w.recording_id) for w in clips}
        if leaked:
            raise AssertionError(f"{level} validation shares sessions with train: {sorted(leaked)[:5]}")
    train_worlds = {r["world"] for r in splits.train}
    leaked_worlds = train_worlds & set(splits.ood_worlds)
    if leaked_worlds:
        raise AssertionError(f"OOD worlds present in train: {sorted(leaked_worlds)}")
