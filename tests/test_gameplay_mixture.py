import math
import unittest

from embodigen.gameplay.mixture import MixtureBuilder, MixtureSpec, diversity, hill_d, inverse_simpson
from embodigen.gameplay.registry import WorldRegistry, WorldSpec
from embodigen.gameplay.splits import SplitConfig, check_ood, split
from embodigen.gameplay.types import Window
from embodigen.gameplay.windows import (
    WindowConfig,
    cap_streams,
    downsample_idle,
    footage_seconds,
    make_windows,
)


def pool(worlds: dict[str, int], recs_per_world: int = 4, turns_per_rec: int = 200) -> list[Window]:
    """``worlds`` maps world -> relative size (recordings scale with it)."""
    cfg = WindowConfig()
    out = []
    for world, size in worlds.items():
        for r in range(recs_per_world * size):
            out += make_windows(f"{world}-{r:03d}", world, f"{world}:s{r % 2}", [False] * turns_per_rec, cfg)
    return out


class DiversityTest(unittest.TestCase):
    def test_hill_number_matches_its_definition(self):
        self.assertAlmostEqual(hill_d({"a": 1.0}), 1.0)
        self.assertAlmostEqual(hill_d({str(i): 1 / 80 for i in range(80)}), 80.0)
        shares = {"a": 0.5, "b": 0.25, "c": 0.25}
        self.assertAlmostEqual(hill_d(shares), math.exp(-sum(p * math.log(p) for p in shares.values())))
        self.assertAlmostEqual(inverse_simpson(shares), 1 / (0.25 + 0.0625 * 2))


class MixtureTest(unittest.TestCase):
    def setUp(self):
        self.windows = pool({"anchor": 3, "b": 1, "c": 2, "d": 1})
        self.builder = MixtureBuilder(self.windows, "anchor", seed=7)

    def test_budgets_are_nested(self):
        for spec in (MixtureSpec("S2", 0.5), MixtureSpec("S5", 0.0), MixtureSpec("S1", 1.0)):
            previous: set[str] = set()
            for hours in (0.05, 0.1, 0.2, 0.4):
                ids = {w.window_id for w in self.builder.select(spec, hours)["windows"]}
                self.assertLessEqual(previous, ids, f"{spec.name} {hours}h")
                previous = ids

    def test_footage_budget_is_met_and_shares_are_exact_in_segments(self):
        sel = self.builder.select(MixtureSpec("S3", 0.271), 0.2)
        self.assertEqual(sel["shortfall_seconds"], 0.0)
        self.assertGreaterEqual(footage_seconds(sel["windows"]), 0.2 * 3600 - 1e-6)
        self.assertLess(footage_seconds(sel["windows"]), 0.2 * 3600 + 4.0 + 1e-6)  # never more than one window over
        self.assertAlmostEqual(sel["anchor_share"], 0.271, delta=1.0 / len(sel["windows"]))

    def test_window_budget_unit_counts_overlaps(self):
        sel = self.builder.select(MixtureSpec("S5", 0.0), 0.1, unit="window")
        self.assertEqual(len(sel["windows"]), 90)  # 360 s / 4 s

    def test_s1_is_one_world_and_s5_has_no_anchor(self):
        s1 = diversity(self.builder.select(MixtureSpec("S1", 1.0), 0.1)["windows"])
        s5 = self.builder.select(MixtureSpec("S5", 0.0), 0.1)["windows"]
        self.assertEqual(s1["worlds"], 1)
        self.assertEqual(s1["hill_d"], 1.0)
        self.assertFalse(any(w.world == "anchor" for w in s5))

    def test_every_mixture_takes_a_prefix_of_each_world(self):
        orders = {w: [x.window_id for x in ws] for w, ws in self.builder.by_world.items()}
        for share in (1.0, 0.514, 0.271, 0.028, 0.0):
            chosen = self.builder.select(MixtureSpec("S", share), 0.15)["windows"]
            for world, order in orders.items():
                ids = [w.window_id for w in chosen if w.world == world]
                self.assertEqual(sorted(ids), sorted(order[: len(ids)]), world)

    def test_non_anchor_prefixes_keep_pool_proportions(self):
        sel = self.builder.select(MixtureSpec("S5", 0.0), 0.3)["windows"]
        counts = {w: sum(1 for x in sel if x.world == w) for w in ("b", "c", "d")}
        # c has twice the data of b and d.
        self.assertAlmostEqual(counts["c"] / counts["b"], 2.0, delta=0.25)
        self.assertAlmostEqual(counts["b"] / counts["d"], 1.0, delta=0.2)

    def test_shortfall_is_reported_not_hidden(self):
        sel = self.builder.select(MixtureSpec("S1", 1.0), 100.0)
        self.assertGreater(sel["shortfall_seconds"], 0)
        self.assertEqual(len(sel["windows"]), len(self.builder.by_world["anchor"]))

    def test_shortfall_keeps_the_share(self):
        for share in (0.514, 0.271, 0.028):
            sel = self.builder.select(MixtureSpec("S", share), 100.0)
            self.assertGreater(sel["shortfall_seconds"], 0)
            self.assertAlmostEqual(sel["anchor_share"], share, delta=2.0 / len(sel["windows"]))

    def test_selection_is_reproducible(self):
        a = MixtureBuilder(self.windows, "anchor", seed=7).select(MixtureSpec("S3", 0.271), 0.2)
        b = MixtureBuilder(list(reversed(self.windows)), "anchor", seed=7).select(MixtureSpec("S3", 0.271), 0.2)
        self.assertEqual([w.window_id for w in a["windows"]], [w.window_id for w in b["windows"]])


class WindowRulesTest(unittest.TestCase):
    def test_windows_are_20_turns_stride_10(self):
        ws = make_windows("r", "w", "s", [False] * 100, WindowConfig())
        self.assertEqual([w.start for w in ws], [0, 10, 20, 30, 40, 50, 60, 70, 80])
        self.assertEqual(footage_seconds(ws), 100 * 0.2)

    def test_idle_downsampling_is_deterministic(self):
        idle = ([True] * 20 + [False] * 20) * 50
        cfg = WindowConfig(idle_keep=0.3)
        ws = make_windows("r", "w", "s", idle, cfg)
        kept = downsample_idle(ws, cfg)
        self.assertEqual(kept, downsample_idle(ws, cfg))
        idle_kept = [w for w in kept if w.idle_frac >= cfg.idle_turn_frac]
        idle_all = [w for w in ws if w.idle_frac >= cfg.idle_turn_frac]
        self.assertLess(len(idle_kept), len(idle_all))
        self.assertTrue(all(w in kept for w in ws if w.idle_frac < cfg.idle_turn_frac))

    def test_stream_cap_keeps_whole_recordings(self):
        rows = [{"recording_id": f"r{i}", "stream": "s0" if i < 6 else "s1", "duration": 600.0} for i in range(8)]
        kept, dropped = cap_streams(rows, WindowConfig(stream_cap_hours=0.5))
        self.assertEqual(sum(r["stream"] == "s0" for r in kept), 3)  # 3 x 10 min fits in 30 min
        self.assertEqual(sum(r["stream"] == "s1" for r in kept), 2)
        self.assertEqual(len(dropped), 3)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.registry = WorldRegistry({
            "anchor": WorldSpec("anchor", "Anchor Craft"),
            "b": WorldSpec("b", "Bee"),
            "c": WorldSpec("c", "Sea", franchise="ocean"),
            "far": WorldSpec("far", "Faraway"),
            "c2": WorldSpec("c2", "Sea Two", franchise="ocean"),
        })
        self.wcfg = WindowConfig()
        self.rows, self.windows = [], {}
        for world, n in (("anchor", 5), ("b", 3), ("c", 3), ("far", 2), ("c2", 2)):
            for i in range(n):
                rid = f"{world}-{i}"
                self.rows.append({"recording_id": rid, "world": world, "stream": f"{world}:0", "duration": 40.0})
                self.windows[rid] = make_windows(rid, world, f"{world}:0", [False] * 200, self.wcfg)

    def test_levels_are_disjoint_from_train(self):
        rows = [r for r in self.rows if r["world"] != "c2"]
        s = split(rows, self.windows, self.registry, "anchor", ["far"], {"far": 1.0}, SplitConfig(clips_per_world=5), self.wcfg)
        train = {r["recording_id"] for r in s.train}
        for level, clips in s.val.items():
            self.assertTrue(clips, level)
            self.assertFalse(train & {w.recording_id for w in clips}, level)
        self.assertFalse(any(r["world"] == "far" for r in s.train))
        # Every pool world keeps at least one recording for training.
        self.assertEqual({r["world"] for r in s.train}, {"anchor", "b", "c"})
        # In-corpus is game-balanced: the same number of clips per world.
        per_world = {}
        for w in s.val["in_corpus"]:
            per_world[w.world] = per_world.get(w.world, 0) + 1
        self.assertEqual(set(per_world.values()), {5})

    def test_source_split_hint_wins_for_the_anchor(self):
        rows = [dict(r, split_hint="val") if r["recording_id"] == "anchor-4" else r for r in self.rows if r["world"] != "c2"]
        s = split(rows, self.windows, self.registry, "anchor", [], {}, SplitConfig(clips_per_world=5), self.wcfg)
        self.assertEqual(s.val_recordings["anchor"], ["anchor-4"])

    def test_ood_world_from_a_training_franchise_is_refused(self):
        with self.assertRaises(SystemExit):
            split(self.rows, self.windows, self.registry, "anchor", ["c2"], {"c2": 1.0}, SplitConfig(), self.wcfg)

    def test_ood_title_below_pass_rate_is_dropped(self):
        rows = [r for r in self.rows if r["world"] != "c2"]
        s = split(rows, self.windows, self.registry, "anchor", ["far"], {"far": 0.5}, SplitConfig(), self.wcfg)
        self.assertEqual(s.ood_worlds, [])
        self.assertTrue(any("pass rate" in n for n in s.notes))


class RegistryTest(unittest.TestCase):
    def test_shipped_registry_resolves_aliases(self):
        reg = WorldRegistry.load()
        self.assertEqual(reg.resolve("我的世界"), "minecraft")
        self.assertEqual(reg.resolve("Minecraft Java Edition"), "minecraft")
        self.assertEqual(reg.resolve("freedoom"), "doom")
        self.assertIsNone(reg.resolve("Tetris"))

    def test_overlap_rules(self):
        a = WorldSpec("a", "Super Plumber", aliases=("SP",), franchise="plumber")
        self.assertEqual(a.overlap(WorldSpec("b", "sp")), "title/alias overlap")
        self.assertIn("franchise", a.overlap(WorldSpec("c", "Plumber Kart", franchise="plumber")))
        self.assertIn("emulated", WorldSpec("d", "X", emulates="nes:x").overlap(WorldSpec("e", "Y", emulates="nes:x")))
        self.assertEqual(a.overlap(WorldSpec("f", "Unrelated")), "")
        with self.assertRaises(SystemExit):
            check_ood(WorldRegistry({"a": a, "g": WorldSpec("g", "Super-Plumber!")}), ["g"], ["a"])


if __name__ == "__main__":
    unittest.main()
