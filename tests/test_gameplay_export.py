"""Gameplay -> embodied export: curated recordings as a LeRobot-shaped dataset."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.actions import KEY_ORDER, decode_action
from embodigen.gameplay.export import ACTION_NAMES, FORMAT, action_vector, export, letterbox, pose
from embodigen.gameplay.pipeline import curate, run_rollouts
from embodigen.gameplay.registry import WorldRegistry
from embodigen.pngio import read_png, write_png


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.registry = WorldRegistry.load()
        raw, cls.curated = cls.tmp / "raw", cls.tmp / "curated"
        run_rollouts(cls.registry, ["raycast-arena"], 2, 8.0, raw, size=(48, 30), workers=2)
        run_rollouts(cls.registry, ["raycast-maze"], 1, 8.0, raw, seed=5, size=(48, 30))
        run_rollouts(cls.registry, ["raycast-rooms"], 1, 8.0, raw, seed=40, profile="afk", size=(48, 30))
        curate(raw, cls.curated, workers=2)
        cls.rows = jsonl(cls.curated / "recordings.jsonl")
        cls.out = cls.tmp / "embodied"
        cls.result = export(cls.curated, cls.out, registry=cls.registry)
        cls.info = json.loads((cls.out / "meta" / "info.json").read_text())
        cls.episodes = jsonl(cls.out / "meta" / "episodes.jsonl")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def data(self, ep: int) -> list[dict]:
        return jsonl(self.out / "data" / "chunk-000" / f"episode_{ep:06d}.jsonl")

    def test_only_certified_recordings_become_episodes(self):
        certified = sorted(r["recording_id"] for r in self.rows if r["passed"])
        rejected = [r["recording_id"] for r in self.rows if not r["passed"]]
        self.assertTrue(rejected, "the afk recording should have been rejected")
        self.assertEqual(sorted(e["recording_id"] for e in self.episodes), certified)
        self.assertEqual(self.info["total_episodes"], len(certified))
        turns = {r["recording_id"]: r["turns"] for r in self.rows if r["passed"]}
        self.assertEqual([e["length"] for e in self.episodes], [turns[e["recording_id"]] for e in self.episodes])
        self.assertEqual(self.info["total_frames"], sum(turns.values()))

    def test_rows_carry_the_curated_actions_unchanged(self):
        for e in self.episodes:
            curated = jsonl(self.curated / "turns" / f"{e['recording_id']}.jsonl")
            rows = self.data(e["episode_index"])
            self.assertEqual(len(rows), len(curated))
            for row, turn in zip(rows, curated):
                packet = decode_action(row["action.text"])
                self.assertEqual((packet.dx, packet.dy, packet.dz), (turn["dx"], turn["dy"], turn["dz"]))
                self.assertEqual([list(k) for k in packet.keys], turn["keys"])
                self.assertEqual(row["action.keys"], turn["keys"])
                vec = row["action"]
                self.assertEqual(len(vec), len(ACTION_NAMES))
                self.assertEqual(vec[:3], [turn["dx"], turn["dy"], turn["dz"]])
                for key in ("w", "a", "mouse_left", "space"):
                    held = sum(key in slot for slot in turn["keys"]) / 6
                    self.assertAlmostEqual(vec[3 + KEY_ORDER.index(key)], held, places=5)
            self.assertTrue(rows[-1]["next.done"])
            self.assertFalse(any(r["next.done"] for r in rows[:-1]))

    def test_indices_are_global_and_tasks_resolve(self):
        index, tasks = 0, {t["task_index"]: t["task"] for t in jsonl(self.out / "meta" / "tasks.jsonl")}
        for e in self.episodes:
            for i, row in enumerate(self.data(e["episode_index"])):
                self.assertEqual((row["index"], row["frame_index"]), (index, i))
                self.assertAlmostEqual(row["timestamp"], i * 0.2, places=5)
                self.assertEqual(tasks[row["task_index"]], row["task"])
                index += 1
        self.assertTrue(all(t.startswith("Play Raycast") for t in tasks.values()))

    def test_engine_truth_is_exported_at_the_frame_size(self):
        features = self.info["features"]
        self.assertEqual(features["observation.images.main"]["shape"], [30, 48, 3])
        self.assertEqual(features["observation.state"]["names"], ["x", "y", "z", "yaw", "pitch"])
        self.assertIn("observation.depth", features)
        self.assertIn("observation.segmentation", features)
        row = self.data(0)[3]
        w, h, c, _ = read_png(self.out / row["frames"]["observation.images.main"])
        self.assertEqual((w, h, c), (48, 30, 3))
        self.assertEqual((self.out / row["frames"]["observation.depth"]).stat().st_size, 48 * 30 * 4)
        self.assertEqual((self.out / row["frames"]["observation.segmentation"]).stat().st_size, 48 * 30 * 2)
        self.assertEqual(row["observation.state"][2], 0.0)
        stats = json.loads((self.out / "meta" / "stats.json").read_text())
        self.assertEqual(len(stats["action"]["mean"]), len(ACTION_NAMES))
        self.assertLessEqual(max(stats["action"]["max"][3:]), 1.0)

    def test_forced_size_letterboxes_and_drops_misaligned_truth(self):
        out = self.tmp / "boxed"
        r = export(self.curated, out, worlds={"raycast-maze"}, registry=self.registry, size=(64, 36))
        self.assertEqual(r["size"], [64, 36])
        self.assertFalse(r["depth"])  # depth pixels would no longer line up with the image
        w, h, _, px = read_png(out / self.data_in(out, 0)[0]["frames"]["observation.images.main"])
        self.assertEqual((w, h), (64, 36))
        self.assertEqual(px[:3], b"\0\0\0")  # 16:10 frames inside a 16:9 box: bars at the sides

    def data_in(self, out: Path, ep: int) -> list[dict]:
        return jsonl(out / "data" / "chunk-000" / f"episode_{ep:06d}.jsonl")

    def test_refuses_to_overwrite_what_it_did_not_write(self):
        with self.assertRaises(SystemExit):
            export(self.curated, self.out, registry=self.registry)  # exists, no --overwrite
        other = self.tmp / "precious"
        other.mkdir()
        (other / "keep.txt").write_text("x")
        with self.assertRaises(SystemExit):
            export(self.curated, other, overwrite=True, registry=self.registry)
        self.assertTrue((other / "keep.txt").exists())

    def test_overwrite_spares_other_lerobot_datasets_and_failed_runs_keep_the_old_export(self):
        foreign = self.tmp / "aloha"
        (foreign / "meta").mkdir(parents=True)
        (foreign / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
        with self.assertRaises(SystemExit):
            export(self.curated, foreign, overwrite=True, registry=self.registry)
        self.assertTrue((foreign / "meta" / "info.json").exists())

        out = self.tmp / "swap"
        export(self.curated, out, worlds={"raycast-maze"}, registry=self.registry)
        self.assertEqual(json.loads((out / "meta" / "info.json").read_text())["format"], FORMAT)
        broken = self.tmp / "curated-broken"
        shutil.copytree(self.curated, broken)
        maze = next(r for r in self.rows if r["world"] == "raycast-maze" and r["passed"])
        raw_frame = Path(json.loads((broken / "ledger.json").read_text())["raw_root"]) / maze["raw_dir"]
        turns = broken / "turns" / f"{maze['recording_id']}.jsonl"
        rows = jsonl(turns)
        rows[5]["frame"] = "frames/missing.png"
        turns.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(OSError):
            export(broken, out, worlds={"raycast-maze"}, overwrite=True, registry=self.registry)
        self.assertTrue((out / "meta" / "info.json").exists(), "the previous export must survive")
        self.assertFalse((self.tmp / ".swap.partial").exists())
        self.assertTrue(raw_frame.exists())
        export(self.curated, out, worlds={"raycast-maze"}, overwrite=True, registry=self.registry)  # and can be redone

    def test_mixed_engine_sizes_letterbox_and_leave_out_depth(self):
        raw, cur = self.tmp / "raw-mixed", self.tmp / "curated-mixed"
        run_rollouts(self.registry, ["raycast-arena"], 1, 4.0, raw, size=(48, 30))
        run_rollouts(self.registry, ["raycast-maze"], 1, 4.0, raw, seed=5, size=(64, 40))
        curate(raw, cur)
        out = self.tmp / "mixed"
        r = export(cur, out, registry=self.registry, width=96)
        self.assertEqual(r["size"], [96, 54])
        self.assertFalse(r["depth"] or r["segmentation"])
        self.assertNotIn("observation.depth", json.loads((out / "meta" / "info.json").read_text())["features"])
        self.assertFalse((out / "images" / "observation.depth").exists())

    def test_video_frames_without_a_decoder_stop_before_writing(self):
        cur = self.tmp / "curated-video"
        shutil.copytree(self.curated, cur)
        turns = cur / "turns" / f"{self.episodes[0]['recording_id']}.jsonl"
        rows = jsonl(turns)
        rows[0]["frame"] = "video.mp4#t=0.000"
        turns.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = self.tmp / "never"
        with self.assertRaises(SystemExit) as ctx:
            export(cur, out, video_backend="", registry=self.registry)
        self.assertIn("inside videos", str(ctx.exception))
        self.assertFalse(out.exists())


class SessionSplitTest(unittest.TestCase):
    def test_a_val_tag_holds_out_the_whole_session(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        registry = WorldRegistry.load()
        raw, cur = tmp / "raw", tmp / "cur"
        run_rollouts(registry, ["raycast-arena"], 3, 4.0, raw, size=(32, 20))
        curate(raw, cur)
        rows = jsonl(cur / "recordings.jsonl")
        for r in rows:  # two parts of one session, only one tagged val; a third, untagged session
            r["session"] = "s1" if r["recording_id"].endswith(("0000", "0001")) else "s2"
            r["split_hint"] = "val" if r["recording_id"].endswith("0000") else ""
        (cur / "recordings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = tmp / "out"
        r = export(cur, out, registry=registry)
        eps = jsonl(out / "meta" / "episodes.jsonl")
        self.assertEqual({e["recording_id"][-4:]: e["split"] for e in eps}, {"0000": "val", "0001": "val", "0002": "train"})
        self.assertEqual(r["splits"], {"train": "0:1", "val": "1:3"})
        self.assertEqual([e["split"] for e in eps], ["train", "val", "val"])


class HelpersTest(unittest.TestCase):
    def test_unlisted_keys_count_once_per_slot(self):
        vec = action_vector(0, 0, 0, [["f13", "f14", "w"]] * 3 + [[]] * 3)
        self.assertAlmostEqual(vec[-1], 0.5)
        self.assertAlmostEqual(vec[3], 0.5)  # w
        self.assertLessEqual(max(action_vector(0, 0, 0, [["f13", "f14", "f15"]] * 6)[3:]), 1.0)

    def test_rgba_frames_at_the_dataset_size_are_converted_not_copied(self):
        from embodigen.gameplay.export import _png_header
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        write_png(tmp / "rgba.png", 4, 2, b"\x10\x20\x30\xff" * 8, channels=4)
        write_png(tmp / "rgb.png", 4, 2, b"\x10\x20\x30" * 8)
        self.assertEqual(_png_header(tmp / "rgba.png"), (4, 2, False))
        self.assertEqual(_png_header(tmp / "rgb.png"), (4, 2, True))

    def test_pose_puts_height_on_z(self):
        self.assertEqual(pose({"pose": [10.0, 64.0, -3.0, 90.0, -5.0]}), [10.0, -3.0, 64.0, 90.0, -5.0])
        self.assertEqual(pose({"pos": [1.0, 2.0], "yaw": 30.0}), [1.0, 2.0, 0.0, 30.0, 0.0])
        self.assertIsNone(pose({"pose": [None, 1, 2, 3, 4]}))
        self.assertIsNone(pose({"gui_open": False}))

    def test_letterbox_keeps_aspect(self):
        red = b"\xff\0\0" * (4 * 2)  # 4x2
        boxed = letterbox(4, 2, red, 4, 4)
        self.assertEqual(len(boxed), 4 * 4 * 3)
        self.assertEqual(boxed[:12], b"\0" * 12)  # top bar
        self.assertEqual(boxed[12:24], red[:12])
        self.assertEqual(letterbox(4, 2, red, 8, 4)[:3], b"\xff\0\0")  # 2x upscale fills the box


if __name__ == "__main__":
    unittest.main()
