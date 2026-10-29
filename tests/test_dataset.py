import json
import tempfile
import unittest
from pathlib import Path

from embodigen.cli import main
from embodigen.modules.pack import read_episode
from embodigen.pipeline import Pipeline, PipelineConfig
from embodigen.preview import render_preview
from embodigen.robots import available_robots


def _config(tmp: str, **overrides) -> PipelineConfig:
    base = dict(out_dir=Path(tmp), resolution=32, fps=6, inject_drift_mm=0.0, max_attempts=1)
    base.update(overrides)
    return PipelineConfig(**base)


class DatasetTest(unittest.TestCase):
    def test_rejected_episode_never_enters_the_dataset(self):
        """A drifted render with no retries left goes to rejected/, not episodes/."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(_config(tmp, inject_drift_mm=25.0, max_attempts=1))
            episode = pipeline.run("把红色方块放进盒子", seed=0)
            self.assertFalse(episode.fidelity.passed)
            self.assertEqual(Path(episode.root).parent.name, "rejected")
            self.assertFalse((Path(tmp) / "episodes" / episode.episode_id).exists())

    def test_rerun_that_passes_clears_the_stale_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            Pipeline(_config(tmp, inject_drift_mm=25.0)).run("把红色方块放进盒子", seed=0)
            episode = Pipeline(_config(tmp)).run("把红色方块放进盒子", seed=0)
            self.assertTrue(episode.fidelity.passed)
            self.assertFalse((Path(tmp) / "rejected" / episode.episode_id).exists())

    def test_batch_writes_one_episode_per_seed_and_an_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = Pipeline(_config(tmp, seed=4)).run_batch("把红色方块放进盒子", count=3, workers=2)
            self.assertEqual([r["seed"] for r in rows], [4, 5, 6])
            index = json.loads((Path(tmp) / "index.json").read_text())
            self.assertEqual(index["summary"]["total"], 3)
            self.assertEqual(index["summary"]["certified"], 3)
            for row in index["episodes"]:  # relative to the dataset, so a copied dataset still resolves
                self.assertEqual(row["path"], f"episodes/{row['episode_id']}")
                self.assertTrue((Path(tmp) / row["path"]).is_dir())

    def test_index_merges_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            Pipeline(_config(tmp, seed=0)).run_batch("把红色方块放进盒子", count=2)
            Pipeline(_config(tmp, seed=1)).run_batch("把红色方块放进盒子", count=2)
            index = json.loads((Path(tmp) / "index.json").read_text())
            self.assertEqual([r["episode_id"] for r in index["episodes"]],
                             ["franka-0000", "franka-0001", "franka-0002"])

    def test_read_episode_keeps_the_robot(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Pipeline(_config(tmp)).run("把红色方块放进盒子", robot_name="mobile")
            scene, _, _ = read_episode(episode.root)
            self.assertEqual(scene.robot, "mobile")
            self.assertEqual(scene.morphology, "mobile")
            fidelity = json.loads((Path(episode.root) / "meta/fidelity.json").read_text())
            self.assertEqual(fidelity["validator"], pipeline_validator_name())

    def test_preview_writes_an_animated_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Pipeline(_config(tmp)).run("把红色方块放进盒子")
            path, frames, width, height = render_preview(
                episode.root, Path(tmp) / "p.gif", scale=1, max_frames=4
            )
            blob = path.read_bytes()
            self.assertEqual(blob[:6], b"GIF89a")
            self.assertEqual(blob[-1:], b"\x3b")
            self.assertEqual(frames, 4)
            self.assertEqual((width, height), (32 * 4 + 2 * 3, 32))

    def test_cli_batch_exit_code_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["gen", "--task", "把红色方块放进盒子", "--out", tmp,
                         "--resolution", "32", "--fps", "6", "-n", "2", "-j", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(main(["index", "--out", tmp]), 0)

    def test_toml_robots_are_discoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "quadruped.toml").write_text('name = "quadruped"\n')
            self.assertIn("quadruped", available_robots(Path(tmp)))
            self.assertIn("franka", available_robots(Path(tmp)))


def pipeline_validator_name() -> str:
    from embodigen.modules.validate import KeypointFidelityValidator

    return KeypointFidelityValidator.name


if __name__ == "__main__":
    unittest.main()
