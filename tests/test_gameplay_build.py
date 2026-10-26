"""End to end on the reference worlds: rollout -> curate -> build."""

import json
import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.actions import decode_action
from embodigen.gameplay.annotate import selected
from embodigen.gameplay.mixture import StudySpec
from embodigen.gameplay.pipeline import build, curate, run_rollouts
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.windows import h01

STUDY = {
    "name": "t",
    "anchor": "raycast-arena",
    "ood": ["raycast-caldera"],
    "budgets_hours": [0.01, 0.02, 0.03],
    "seed": 11,
    "windows": {"stream_cap_hours": 0.05},
    "val": {"clips_per_world": 4},
    "reasoning": {"rate": 0.1},
    "serialize": {"shard_size": 8},
    "mixtures": [
        {"name": "S1", "anchor_share": 1.0},
        {"name": "S3", "anchor_share": 0.271},
        {"name": "S5", "anchor_share": 0.0},
    ],
}


def shard_rows(directory: Path, prefix: str = "train") -> list[dict]:
    rows = []
    for path in sorted(directory.glob(f"{prefix}-*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.registry = WorldRegistry.load()
        raw = root / "raw"
        worlds = ["raycast-arena", "raycast-maze", "raycast-rooms", "raycast-caldera"]
        run_rollouts(cls.registry, worlds[:1], 5, 20.0, raw, size=(48, 30), gt=False, workers=4)
        run_rollouts(cls.registry, worlds[1:], 3, 20.0, raw, size=(48, 30), gt=False, workers=4)
        run_rollouts(cls.registry, ["raycast-rooms"], 1, 20.0, raw, seed=40, profile="afk", size=(48, 30), gt=False)
        cls.curated = root / "curated"
        cls.cur_ledger = curate(raw, cls.curated, workers=4)
        cls.study = StudySpec.from_toml(STUDY)
        cls.out = root / "mixes"
        cls.ledger = build(cls.curated, cls.study, cls.registry, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rejected_recordings_never_reach_a_mixture(self):
        rejected = [json.loads(line)["recording_id"] for line in (self.curated / "rejected.jsonl").read_text().splitlines()]
        self.assertIn("raycast-rooms-0040-afk", rejected)
        self.assertFalse((self.curated / "turns" / "raycast-rooms-0040-afk.jsonl").exists())
        for g in self.ledger["grid"]:
            ids = (self.out / g["path"] / "windows.txt").read_text().split()
            self.assertFalse(any(i.startswith("raycast-rooms-0040-afk") for i in ids))

    def test_budgets_nest_on_disk(self):
        for mix in ("S1", "S3", "S5"):
            sets = [set((self.out / mix / f"{b:g}h" / "windows.txt").read_text().split()) for b in (0.01, 0.02, 0.03)]
            self.assertLessEqual(sets[0], sets[1])
            self.assertLessEqual(sets[1], sets[2])

    def test_validation_is_disjoint_and_ood_is_unseen(self):
        val_recs = {r["recording"] for level in ("anchor", "in_corpus", "ood") for r in shard_rows(self.out / "val", level)}
        for g in self.ledger["grid"]:
            train = shard_rows(self.out / g["path"])
            self.assertFalse(val_recs & {r["recording"] for r in train}, g["path"])
            self.assertFalse(any(r["world"] == "raycast-caldera" for r in train))
        ood = shard_rows(self.out / "val", "ood")
        self.assertTrue(ood)
        self.assertEqual({r["world"] for r in ood}, {"raycast-caldera"})

    def test_mixture_shares(self):
        grid = {(g["mixture"], g["budget_hours"]): g for g in self.ledger["grid"]}
        self.assertEqual(grid[("S1", 0.02)]["hill_d"], 1.0)
        self.assertEqual(grid[("S5", 0.02)]["anchor_share"], 0.0)
        s3 = grid[("S3", 0.03)]
        self.assertAlmostEqual(s3["anchor_share"], 0.271, delta=1.0 / s3["samples"] + 1e-9)
        self.assertGreater(s3["hill_d"], 1.5)

    def test_samples_follow_the_template_and_reference_real_frames(self):
        media = Path(self.ledger["media_root"])
        rows = shard_rows(self.out / "S3" / "0.03h")
        self.assertTrue(rows)
        for row in rows[:5]:
            self.assertEqual(len(row["messages"]), 40)
            self.assertEqual(len(row["images"]), 20)
            for msg in row["messages"][1::2]:
                self.assertTrue(msg["content"].startswith("<think>\n"))
                decode_action(msg["content"])
            for img in row["images"][:3]:
                self.assertTrue((media / img).exists(), img)

    def test_rationales_are_per_turn_not_per_mixture(self):
        texts: dict[tuple[str, int], str] = {}
        for g in self.ledger["grid"]:
            for row in shard_rows(self.out / g["path"]):
                for k, msg in enumerate(row["messages"][1::2]):
                    turn = row["start_turn"] + k
                    has = msg["content"].count("\n") > 2
                    self.assertEqual(has, selected(row["recording"], turn, 0.1, "think:11"))
                    if has:
                        key = (row["recording"], turn)
                        self.assertEqual(texts.setdefault(key, msg["content"]), msg["content"])

    def test_manifests_hash_their_shards(self):
        import hashlib

        manifest = json.loads((self.out / "S3" / "0.02h" / "manifest.json").read_text())
        for shard in manifest["shards"]:
            blob = (self.out / "S3" / "0.02h" / shard["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), shard["sha256"])
        self.assertEqual(sum(s["samples"] for s in manifest["shards"]), manifest["samples"])

    def test_rebuild_is_identical_and_leaves_no_stale_shards(self):
        before = {g["path"]: json.loads((self.out / g["path"] / "manifest.json").read_text())["windows_sha256"]
                  for g in self.ledger["grid"]}
        study = StudySpec.from_toml({**STUDY, "serialize": {"shard_size": 1000}})
        ledger = build(self.curated, study, self.registry, self.out)
        self.assertEqual(ledger["checks"]["rationale_cache"]["misses"], 0)
        for path, digest in before.items():
            manifest = json.loads((self.out / path / "manifest.json").read_text())
            self.assertEqual(manifest["windows_sha256"], digest)
            self.assertEqual(len(list((self.out / path).glob("train-*.jsonl"))), 1)
        # Restore the original layout for any test that runs after this one.
        build(self.curated, self.study, self.registry, self.out)

    def test_hash_helper_is_stable(self):
        self.assertEqual(h01("a", 1), h01("a", 1))
        self.assertNotEqual(h01("a", 1), h01("a", 2))


if __name__ == "__main__":
    unittest.main()
