"""Data mixtures with nested budgets, and how diverse they are.

A scaling study only means something if each run differs from its
neighbours in the one variable under study. Two properties make that true
here by construction rather than by care:

* **Budgets are nested.** Every world's windows have one fixed priority
  order, and non-anchor worlds are merged into one interleaved order whose
  prefixes keep the pool's proportions. A budget takes a prefix of each, so
  the 50 h set is inside the 100 h set, which is inside the 200 h set ...
* **Mixtures are nested.** All mixtures draw from those same orders; they only
  move the cut between the anchor prefix and the non-anchor prefix.

Diversity is reported as GameScaling reports it: the Hill number of order
one, ``D = exp(-sum p_w log p_w)`` over the share of segments per world (1 for
a single world, N for N equally weighted worlds), with inverse Simpson and the
largest share alongside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from embodigen.gameplay.types import Window
from embodigen.gameplay.windows import footage_seconds, h01


@dataclass(frozen=True)
class MixtureSpec:
    name: str
    anchor_share: float  # share of gameplay segments from the anchor world
    temperature: float = 1.0  # non-anchor weights ~ availability ** temperature
    exclude: tuple[str, ...] = ()


def hill_d(shares: dict[str, float]) -> float:
    return math.exp(-sum(p * math.log(p) for p in shares.values() if p > 0))


def inverse_simpson(shares: dict[str, float]) -> float:
    s = sum(p * p for p in shares.values())
    return 1.0 / s if s > 0 else 0.0


def shares_of(windows: list[Window]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for w in windows:
        counts[w.world] = counts.get(w.world, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


def diversity(windows: list[Window]) -> dict[str, Any]:
    shares = shares_of(windows)
    largest = max(shares.items(), key=lambda kv: kv[1]) if shares else ("", 0.0)
    return {
        "worlds": len(shares),
        "hill_d": round(hill_d(shares), 3),
        "inverse_simpson": round(inverse_simpson(shares), 3),
        "largest_world": largest[0],
        "largest_share": round(largest[1], 4),
        "shares": {k: round(v, 5) for k, v in shares.items()},
    }


class MixtureBuilder:
    """Fixed priority orders over a training pool; mixtures are cuts through them."""

    def __init__(self, pool: list[Window], anchor: str, seed: int = 0) -> None:
        self.anchor = anchor
        self.seed = seed
        by_world: dict[str, list[Window]] = {}
        for w in pool:
            by_world.setdefault(w.world, []).append(w)
        self.by_world = {
            world: sorted(ws, key=lambda w: h01("order", seed, w.window_id))
            for world, ws in sorted(by_world.items())
        }

    def available(self) -> dict[str, int]:
        return {w: len(ws) for w, ws in self.by_world.items()}

    def _interleaved(self, spec: MixtureSpec) -> list[Window]:
        """Non-anchor windows in one order whose every prefix keeps the target weights."""
        worlds = [w for w in self.by_world if w != self.anchor and w not in spec.exclude]
        weights = {w: len(self.by_world[w]) ** spec.temperature for w in worlds}
        keyed = []
        for world in worlds:
            step = 1.0 / weights[world]
            for rank, win in enumerate(self.by_world[world]):
                keyed.append(((rank + 0.5) * step, h01("tie", self.seed, win.window_id), win))
        keyed.sort(key=lambda k: (k[0], k[1]))
        return [k[2] for k in keyed]

    @staticmethod
    def _seconds(windows: list[Window], unit: str) -> float:
        if unit == "window":
            return sum(w.turns for w in windows) * 0.2
        return footage_seconds(windows)

    def select(self, spec: MixtureSpec, budget_hours: float, unit: str = "footage") -> dict[str, Any]:
        """The smallest ``N`` segments, split ``anchor_share`` : rest by count,
        whose gameplay reaches the budget.

        Both the anchor cut and the non-anchor cut grow monotonically with
        ``N``, and ``N`` grows with the budget, so budgets stay nested while
        shares are exact in segments (the unit the paper reports them in).
        """
        target = budget_hours * 3600.0
        share = spec.anchor_share if self.anchor not in spec.exclude else 0.0
        anchor_order = self.by_world.get(self.anchor, []) if share > 0 else []
        other_order = self._interleaved(spec) if share < 1 else []

        def cut(n: int) -> list[Window]:
            na = min(int(round(n * share)), len(anchor_order))
            no = min(n - int(round(n * share)), len(other_order))
            return anchor_order[:na] + other_order[:no]

        # The largest N either side can supply at this share. When the pool
        # runs out the mixture shrinks; it never tops up from the other side,
        # which would quietly turn S2/S3/S4 into the same mixture.
        caps = []
        if share > 0:
            caps.append(int(len(anchor_order) / share))
        if share < 1:
            caps.append(int(len(other_order) / (1 - share)))
        hi = min(caps) if caps else 0
        lo = 0
        if self._seconds(cut(hi), unit) < target:
            lo = hi  # short of data: keep the share, report the shortfall
        while lo < hi:
            mid = (lo + hi) // 2
            if self._seconds(cut(mid), unit) >= target - 1e-9:
                hi = mid
            else:
                lo = mid + 1
        chosen = cut(lo)
        got = self._seconds(chosen, unit)
        n_anchor = sum(1 for w in chosen if w.world == self.anchor)
        return {
            "windows": chosen,
            "target_seconds": target,
            "seconds": round(got, 3),
            "shortfall_seconds": round(max(0.0, target - got), 3),
            "anchor_share_target": spec.anchor_share,
            "anchor_share": round(n_anchor / len(chosen), 4) if chosen else 0.0,
        }


@dataclass
class StudySpec:
    """Everything a scaling grid needs, read from ``configs/studies/*.toml``."""

    name: str
    anchor: str
    budgets_hours: list[float]
    mixtures: list[MixtureSpec]
    ood: list[str] = field(default_factory=list)
    vl_anchor_share: float = 0.15
    seed: int = 3407
    windows: dict[str, Any] = field(default_factory=dict)
    val: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=dict)
    serialize: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_toml(data: dict[str, Any]) -> "StudySpec":
        mixtures = [
            MixtureSpec(
                name=m["name"],
                anchor_share=float(m["anchor_share"]),
                temperature=float(m.get("temperature", 1.0)),
                exclude=tuple(m.get("exclude", ())),
            )
            for m in data.get("mixtures", [])
        ]
        if not mixtures:
            raise SystemExit("study defines no [[mixtures]]")
        budgets = sorted(float(b) for b in data["budgets_hours"])
        return StudySpec(
            name=data["name"],
            anchor=data.get("anchor", ""),
            budgets_hours=budgets,
            mixtures=mixtures,
            ood=list(data.get("ood", [])),
            vl_anchor_share=float(data.get("vl_anchor_share", 0.15)),
            seed=int(data.get("seed", 3407)),
            windows=dict(data.get("windows", {})),
            val=dict(data.get("val", {})),
            reasoning=dict(data.get("reasoning", {})),
            serialize=dict(data.get("serialize", {})),
        )
