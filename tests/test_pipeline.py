import json
import tempfile
import unittest
from pathlib import Path

from embodigen.modules.pack import read_episode
from embodigen.pipeline import Pipeline, PipelineConfig


class PipelineTest(unittest.TestCase):
    def test_packs_a_self_contained_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(
                PipelineConfig(
                    out_dir=Path(tmp), resolution=32, fps=6, max_attempts=1, inject_drift_mm=0.0
                )
            )
            episode = pipeline.run("把红色方块放进盒子", seed=1)
            root = Path(episode.root)

            self.assertTrue((root / "meta/info.json").exists())
            self.assertTrue((root / "meta/provenance.json").exists())
            self.assertTrue((root / "meta/fidelity.json").exists())
            self.assertTrue((root / "data/chunk-000/episode_000000.jsonl").exists())
            self.assertEqual(len(list((root / "frames/photoreal").glob("*.png"))), len(episode.gt))

            info = json.loads((root / "meta/info.json").read_text())
            self.assertEqual(info["robot_type"], "franka")
            self.assertEqual(info["total_frames"], len(episode.gt))

    def test_drift_is_rejected_and_retried(self):
        """The closed loop: a drifted render fails, the retry is certified."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(
                PipelineConfig(
                    out_dir=Path(tmp),
                    resolution=32,
                    fps=6,
                    threshold_mm=12.0,
                    max_attempts=3,
                    inject_drift_mm=25.0,
                )
            )
            episode = pipeline.run("把红色方块放进盒子", seed=0)
            self.assertTrue(episode.fidelity.passed, episode.fidelity.notes)
            self.assertEqual(episode.provenance.attempts, 2)
            self.assertEqual(len(episode.provenance.seeds), 2)
            # Retries lean harder on geometry, per the design doc.
            self.assertGreater(episode.provenance.condition_weights["geometry"], 1.0)

    def test_undrifted_episode_passes_on_the_first_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(
                PipelineConfig(
                    out_dir=Path(tmp), resolution=32, fps=6, max_attempts=3, inject_drift_mm=0.0
                )
            )
            episode = pipeline.run("把蓝色方块放进盒子", seed=0)
            self.assertTrue(episode.fidelity.passed)
            self.assertEqual(episode.provenance.attempts, 1)
            self.assertLess(episode.fidelity.p95_error_mm, 12.0)

    def test_packed_episode_revalidates_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(
                PipelineConfig(
                    out_dir=Path(tmp), resolution=32, fps=6, max_attempts=1, inject_drift_mm=0.0
                )
            )
            episode = pipeline.run("把红色方块放进盒子", seed=2)
            scene, gt, rendered = read_episode(episode.root)
            report = pipeline.validator.score(rendered, gt, scene, 12.0)
            self.assertTrue(report.passed)
            self.assertAlmostEqual(report.p95_error_mm, episode.fidelity.p95_error_mm, places=6)


if __name__ == "__main__":
    unittest.main()
