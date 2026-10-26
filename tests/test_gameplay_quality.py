import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.actions import Canonicalizer
from embodigen.gameplay.envs.raycast import RaycastWorld
from embodigen.gameplay.quality import (
    QualityConfig,
    assess,
    capture_metrics,
    capture_reasons,
    check_alignment,
)
from embodigen.gameplay.recording import read_recording
from embodigen.gameplay.rollout import RolloutSpec, rollout
from embodigen.gameplay.types import FOCUS, KEY, MOVE, InputEvent, Observation, Recording, RecordingMeta


def play(tmp: str, name: str, profile: str = "player", lag_ms: float = 0.0, seed: int = 1, layout: str = "arena",
         seconds: float = 15.0, drift: float = 0.0):
    env = RaycastWorld(64, 40, layout=layout)
    spec = RolloutSpec(world="raycast-arena", engine="raycast", engine_args={}, seconds=seconds, seed=seed,
                       profile=profile, lag_ms=lag_ms, drift_ms_per_min=drift, width=64, height=40, gt=False)
    rollout(env, spec, Path(tmp), name)
    return read_recording(Path(tmp) / name), Path(tmp) / name


def gate(rec, root, cfg=None):
    cfg = cfg or QualityConfig()
    alignment, events, corrected = check_alignment(rec, root, cfg)
    turns = Canonicalizer().turns(events, rec.observations, rec.meta.duration)
    return assess(rec, turns, alignment, corrected, cfg)


class AlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.clean = play(cls.tmp.name, "clean")
        cls.late = play(cls.tmp.name, "late", lag_ms=300.0)
        cls.early = play(cls.tmp.name, "early", lag_ms=-200.0, layout="maze", seed=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clean_rollout_is_aligned(self):
        report = gate(*self.clean)
        self.assertTrue(report.passed, report.reasons)
        self.assertEqual(report.alignment["status"], "aligned")
        self.assertEqual(report.realigned_ms, 0.0)

    def test_late_input_log_is_measured_and_corrected(self):
        report = gate(*self.late)
        self.assertTrue(report.passed, report.reasons)
        # Measured from pixels and events only, to within one 33 ms frame.
        self.assertAlmostEqual(report.realigned_ms, 300.0, delta=34.0)
        self.assertEqual(report.alignment["status"], "aligned")

    def test_early_input_log_is_measured_too(self):
        report = gate(*self.early)
        self.assertAlmostEqual(report.realigned_ms, -200.0, delta=34.0)

    def test_without_realign_a_desync_is_rejected(self):
        report = gate(*self.late, cfg=QualityConfig(realign=False))
        self.assertFalse(report.passed)
        self.assertTrue(any(r.startswith("desync") for r in report.reasons), report.reasons)

    def test_alignment_never_reads_the_injected_lag(self):
        rec, root = self.late
        rec.meta.injected_lag_ms = 0.0  # the audit field is not an input to the checker
        self.assertAlmostEqual(gate(rec, root).realigned_ms, 300.0, delta=34.0)


class DriftTest(unittest.TestCase):
    def test_clock_drift_is_rejected_not_shifted(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = gate(*play(tmp, "drift", seconds=60.0, drift=600.0, seed=4))
            steady = gate(*play(tmp, "steady", seconds=60.0, lag_ms=200.0, seed=4))
        # Drift is never "fixed" with one constant shift: the recording stays out.
        self.assertFalse(report.passed, report.alignment)
        self.assertEqual(report.realigned_ms, 0.0)
        self.assertTrue(any(r.startswith(("clock_drift", "desync")) for r in report.reasons), report.reasons)
        self.assertEqual(len(report.alignment["segments"]), 3)
        # A constant offset over the same length is still one correctable lag.
        self.assertTrue(steady.passed, steady.reasons)
        self.assertAlmostEqual(steady.realigned_ms, 200.0, delta=34.0)


    def test_mixed_motion_segments_are_summarised(self):
        # Moves only in the first half: one segment measurable, one static.
        with tempfile.TemporaryDirectory() as tmp:
            rec, root = play(tmp, "half", seconds=30.0, seed=6)
            rec.events = [e for e in rec.events if e.t < 12.0]
            alignment, _, _ = check_alignment(rec, root, QualityConfig())
        self.assertEqual(len(alignment["segments"]), 2)
        self.assertEqual(alignment["frames"], 900)
        self.assertIn(alignment["status"], ("aligned", "inconclusive"))


class CaptureRulesTest(unittest.TestCase):
    def test_afk_player_is_rejected_for_idle_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = gate(*play(tmp, "afk", profile="afk", seed=5))
        self.assertFalse(report.passed)
        self.assertTrue(any(r.startswith("idle_windows") for r in report.reasons), report.reasons)

    def test_alt_tabbing_player_is_rejected_for_non_game_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, root = play(tmp, "away", profile="distracted", seed=2)
        # Force one long alt-tab so the test does not depend on the dice.
        rec.events.append(InputEvent(t=2.0, kind=FOCUS, code="browser"))
        rec.events += [InputEvent(t=2.0 + 0.01 * i, kind=KEY, code="x", down=i % 2 == 0) for i in range(300)]
        rec.events.append(InputEvent(t=6.0, kind=FOCUS, code="game"))
        report = gate(rec, root)
        self.assertTrue(any(r.startswith("non_game_input") for r in report.reasons), report.reasons)

    def _synthetic(self, events, seconds=60.0):
        meta = RecordingMeta(recording_id="syn", world="w", source="test", stream="s", duration=seconds,
                             frame_hz=5.0, width=200, height=100)
        obs = [Observation(t=k / 5.0, index=k, frame=f"video.mp4#t={k / 5.0:.3f}") for k in range(int(seconds * 5))]
        rec = Recording(meta=meta, events=events, observations=obs)
        turns = Canonicalizer().turns(events, obs, seconds)
        return capture_reasons(capture_metrics(rec, turns, QualityConfig()), QualityConfig())

    def test_cursor_recentering_is_rejected(self):
        events = []
        for i in range(600):
            t = i * 0.1
            events.append(InputEvent(t=t, kind=MOVE, dx=5.0, dy=0.0, x=100.0, y=50.0))  # warped to centre
            events.append(InputEvent(t=t + 0.01, kind=KEY, code="w", down=i % 2 == 0))
        reasons = self._synthetic(events)
        self.assertTrue(any(r.startswith("cursor_recentering") for r in reasons), reasons)

    def test_sparse_input_is_rejected(self):
        events = [InputEvent(t=float(i), kind=MOVE, dx=3.0) for i in range(30)]  # 30 events/min
        reasons = self._synthetic(events)
        self.assertTrue(any(r.startswith("sparse_input") for r in reasons), reasons)
        self.assertTrue(any(r.startswith("sparse_keys") for r in reasons), reasons)

    def test_video_only_recordings_skip_alignment(self):
        meta = RecordingMeta(recording_id="v", world="w", source="vpt", stream="s", duration=10.0,
                             frame_hz=20.0, width=640, height=360)
        obs = [Observation(t=k / 20.0, index=k, frame=f"video.mp4#t={k / 20.0:.3f}") for k in range(200)]
        alignment, _, _ = check_alignment(Recording(meta, [], obs), Path("."), QualityConfig())
        self.assertEqual(alignment["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
