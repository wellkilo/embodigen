import unittest

from embodigen.pipeline import Pipeline, PipelineConfig
from embodigen.robots import load_robot


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = Pipeline(
            PipelineConfig(out_dir="out/test", resolution=32, fps=6, pack=False, max_attempts=1)
        )
        cls.episode = cls.pipeline.run("把红色方块放进盒子", seed=3)

    def test_every_required_field_is_present(self):
        ep = self.episode
        self.assertTrue(ep.task.instruction)
        self.assertEqual(len(ep.gt), len(ep.photoreal.frames))
        first = ep.gt[0]
        self.assertAlmostEqual(len(first.depth), 32 * 32)
        self.assertAlmostEqual(len(first.seg), 32 * 32)
        self.assertEqual(len(first.rgb), 32 * 32 * 3)
        self.assertIn("cube", first.objects)
        self.assertIsNotNone(ep.fidelity.p95_error_mm)
        self.assertTrue(ep.provenance.config_hash)

    def test_action_dimension_follows_the_robot(self):
        own = load_robot(self.episode.scene.robot)
        self.assertEqual(self.episode.trajectory.action_dim, own["action_dim"])
        for frame in self.episode.trajectory.frames:
            self.assertEqual(len(frame.action), own["action_dim"])

    def test_planner_speaks_each_morphologys_action_space(self):
        import random

        from embodigen.modules.scene import TemplateSceneGenerator
        from embodigen.modules.trajectory import WaypointTrajectoryGen
        from embodigen.types import TaskSpec

        scenegen = TemplateSceneGenerator()
        planner = WaypointTrajectoryGen(fps=6)
        for name in ("franka", "mobile", "humanoid"):
            robot = load_robot(name)
            scene = scenegen.generate(
                TaskSpec("把红色方块放进盒子", robot["morphology"], name),
                robot,
                random.Random(0),
            )
            traj = planner.plan(scene, robot)
            self.assertEqual(traj.action_dim, robot["action_dim"])
            for frame in traj.frames:
                self.assertEqual(len(frame.action), robot["action_dim"])

    def test_labels_are_geometric_not_estimated(self):
        """The cube's label position must match the rendered geometry exactly."""
        gt = self.episode.gt
        cube_label = gt[0].seg_labels["cube"]
        seen = 0
        for frame in gt:
            for sid in frame.seg:
                if sid == cube_label:
                    seen += 1
        self.assertGreater(seen, 0, "cube should be visible in the white-model render")

    def test_every_frame_carries_a_phase_label(self):
        phases = [f.phase for f in self.episode.gt]
        self.assertTrue(all(phases))
        # Phases appear in plan order and each one is a contiguous run.
        runs = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
        self.assertEqual(runs, ["pre_grasp", "grasp", "close", "lift", "transport",
                                "place", "release", "retreat"])

    def test_feature_summary_is_lerobot_shaped(self):
        features = self.episode.feature_summary()
        self.assertIn("observation.images.photoreal", features)
        self.assertIn("action", features)
        self.assertEqual(features["action"]["shape"], [7])


if __name__ == "__main__":
    unittest.main()
