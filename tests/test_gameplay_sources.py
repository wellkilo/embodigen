import json
import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.pipeline import curate, run_ingest
from embodigen.gameplay.recording import read_recording
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.sources.vpt import stream_of
from embodigen.gameplay.types import BUTTON, KEY, MOVE

STEM = "cheeky-cornflower-setter-02e496ce4abb-20220421-092639"


def vpt_step(i: int, stuck: bool = False) -> dict:
    """A synthetic line in the VPT contractor jsonl shape (no real data shipped)."""
    keys = ["key.keyboard.w"] if i % 40 < 30 else ["key.keyboard.w", "key.keyboard.space"]
    if i % 100 == 50:
        keys.append("key.keyboard.left.shift")
    buttons = [0] if (stuck and i < 5) or (40 <= i % 60 < 44) else []
    new_buttons = [0] if (stuck and i == 0) or i % 60 == 40 else []
    return {
        "mouse": {"x": 640.0, "y": 360.0, "dx": float((i % 7) - 3), "dy": float((i % 5) - 2),
                  "buttons": buttons, "newButtons": new_buttons, "dwheel": 0},
        "keyboard": {"keys": keys, "newKeys": [], "chars": ""},
        "isGuiOpen": i % 200 > 190,
        "hotbar": (i // 50) % 9,
    }


def write_vpt(root: Path, stem: str, frames: int, stuck: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stem}.jsonl"
    path.write_text("".join(json.dumps(vpt_step(i, stuck)) + "\n" for i in range(frames)), encoding="utf-8")
    (root / f"{stem}.mp4").write_bytes(b"not really a video")
    return path


class VPTSourceTest(unittest.TestCase):
    def test_converts_frames_keys_buttons_and_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, raw = Path(tmp) / "vpt", Path(tmp) / "raw"
            write_vpt(src / "train", STEM, 600, stuck=True)
            rows = run_ingest("vpt", src, raw, WorldRegistry.load())
            self.assertEqual(len(rows), 1)
            rec = read_recording(raw / f"vpt-{STEM}")

            self.assertEqual(rec.meta.world, "minecraft")
            self.assertEqual(rec.meta.stream, "vpt:cheeky-cornflower-setter")
            self.assertEqual(rec.meta.duration, 30.0)
            self.assertEqual(len(rec.observations), 600)
            self.assertEqual(rec.observations[20].frame, "video.mp4#t=1.000")
            self.assertTrue((raw / f"vpt-{STEM}" / "video.mp4").is_symlink())

            first_attack = min(e.t for e in rec.events if e.kind == BUTTON and e.code == "mouse_left" and e.down)
            # The attack "held" on frame 0 is VPT's stuck-button artefact, not a click.
            self.assertEqual(first_attack, 2.0)
            codes = {e.code for e in rec.events if e.kind == KEY}
            self.assertEqual(codes, {"w", "space", "shift"})
            self.assertTrue(any(e.kind == MOVE for e in rec.events))
            self.assertTrue(all(e.x is None for e in rec.events if e.kind == MOVE))

    def test_test_directory_marks_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, raw = Path(tmp) / "vpt", Path(tmp) / "raw"
            write_vpt(src / "test", STEM, 200)
            rows = run_ingest("vpt", src, raw, WorldRegistry.load())
            self.assertEqual(rows[0]["split_hint"], "val")

    def test_ingested_recordings_curate_without_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, raw, cur = Path(tmp) / "vpt", Path(tmp) / "raw", Path(tmp) / "curated"
            write_vpt(src, STEM, 600)
            run_ingest("vpt", src, raw, WorldRegistry.load())
            ledger = curate(raw, cur)
            row = json.loads((cur / "recordings.jsonl").read_text().splitlines()[0])
            self.assertTrue(row["passed"], row["reasons"])
            self.assertEqual(row["alignment"]["status"], "skipped")
            self.assertEqual(row["turns"], 150)
            self.assertEqual(ledger["worlds"]["minecraft"]["certified"], 1)
            turn = json.loads((cur / row["turns_file"]).read_text().splitlines()[5])
            self.assertTrue(turn["frame"].startswith("video.mp4#t="))

    def test_stream_name_falls_back_to_the_stem(self):
        self.assertEqual(stream_of("some-odd-name"), "vpt:some-odd-name")


if __name__ == "__main__":
    unittest.main()
