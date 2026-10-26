"""Alignment on frames kept inside a video, and lag tracking past the search window.

The video tests re-encode a reference-world rollout into an MP4 and point the
recording at ``video.mp4#t=...``, so the same lag must be found from decoded
video as from the PNGs. They need the [video] extra (PyAV) and skip without it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.quality import QualityConfig, check_alignment
from embodigen.gameplay.recording import read_recording
from embodigen.gameplay.video import split_ref, video_reader
from embodigen.pngio import read_png
from tests.test_gameplay_quality import gate, play

try:
    import av

    HAS_AV = True
except ImportError:
    HAS_AV = False


def to_video(root: Path) -> None:
    """Encode the rollout's PNG frames into video.mp4 and re-point obs.jsonl at it."""
    rows = [json.loads(line) for line in (root / "obs.jsonl").read_text().splitlines()]
    w, h, _, _ = read_png(root / rows[0]["frame"])
    with av.open(str(root / "video.mp4"), "w") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"qscale": "2"}
        for k, row in enumerate(rows):
            _, _, c, px = read_png(root / row["frame"])
            frame = av.VideoFrame(w, h, "rgb24")
            frame.planes[0].update(px if c == 3 else bytes(b for i in range(w * h) for b in px[i * c : i * c + 3]))
            frame.pts = k
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    for k, row in enumerate(rows):
        row["frame"] = f"video.mp4#t={k / 30:.3f}"
    (root / "obs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class FrameRefTest(unittest.TestCase):
    def test_split_ref(self):
        self.assertEqual(split_ref("video.mkv#t=12.345"), ("video.mkv", 12.345))
        self.assertEqual(split_ref("frames/000001.png"), ("frames/000001.png", None))

    def test_disabled_backend_skips_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, root = play(tmp, "v", seconds=8.0)
            rows = [dict(json.loads(line), frame=f"video.mp4#t={i / 30:.3f}")
                    for i, line in enumerate((root / "obs.jsonl").read_text().splitlines())]
            (root / "obs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "video.mp4").write_bytes(b"")
            alignment, _, _ = check_alignment(read_recording(root), root, QualityConfig(video_backend=""))
        self.assertEqual(alignment["status"], "skipped")
        self.assertIn("disabled", alignment["reason"])
        self.assertEqual(video_reader("nope")[0], None)


@unittest.skipUnless(HAS_AV, "needs the [video] extra (PyAV)")
class VideoAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.recs = {}
        for name, lag in (("clean", 0.0), ("late", 300.0)):
            _, root = play(cls.tmp.name, name, lag_ms=lag, seconds=15.0)
            to_video(root)
            cls.recs[name] = (read_recording(root), root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frames_now_come_from_the_video(self):
        rec, _ = self.recs["clean"]
        self.assertTrue(all(o.frame.startswith("video.mp4#t=") for o in rec.observations))

    def test_clean_video_is_aligned(self):
        report = gate(*self.recs["clean"])
        self.assertEqual(report.alignment["status"], "aligned", report.alignment)
        self.assertEqual(report.realigned_ms, 0.0)

    def test_lag_is_measured_from_decoded_video(self):
        report = gate(*self.recs["late"])
        self.assertTrue(report.passed, report.reasons)
        self.assertAlmostEqual(report.realigned_ms, 300.0, delta=34.0)


class LagTrackingTest(unittest.TestCase):
    def test_drift_past_the_search_window_is_followed_and_rejected(self):
        # 400 ms/min over 3 min: the last segment sits ~1.1 s late, outside
        # the +-600 ms search. It is reached by re-centring on the lag found
        # nearer the start, and the spread marks the recording as drift.
        with tempfile.TemporaryDirectory() as tmp:
            report = gate(*play(tmp, "slow", seconds=180.0, drift=400.0, seed=4))
        self.assertFalse(report.passed)
        self.assertEqual(report.alignment["status"], "drift", report.alignment)
        last = report.alignment["segments"][-1]
        self.assertGreater(last["lag_ms"], QualityConfig().max_lag_ms)
        self.assertTrue(last.get("tracked"))
        self.assertTrue(any(r.startswith("clock_drift") for r in report.reasons))


if __name__ == "__main__":
    unittest.main()
