import random
import unittest

from embodigen.geometry import dist
from embodigen.modules.scene import TemplateSceneGenerator
from embodigen.robots import load_robot
from embodigen.types import TaskSpec


class SceneTest(unittest.TestCase):
    def setUp(self):
        self.robot = load_robot("franka")
        self.gen = TemplateSceneGenerator()

    def test_instruction_colour_reaches_the_scene(self):
        scene = self.gen.generate(
            TaskSpec("把蓝色方块放进盒子"), self.robot, random.Random(0)
        )
        cube = scene.object("cube")
        self.assertGreater(cube.albedo[2], cube.albedo[0], "blue should dominate red")

    def test_layout_respects_reach_and_clearance(self):
        for seed in range(20):
            scene = self.gen.generate(
                TaskSpec("把红色方块放进盒子", seed=seed), self.robot, random.Random(seed)
            )
            cube = scene.object("cube")
            bin_floor = scene.object("bin_floor")
            self.assertLess(dist(cube.center, (0.0, 0.0, 0.0)), self.robot["reach"])
            self.assertLess(dist(bin_floor.center, (0.0, 0.0, 0.0)), self.robot["reach"])
            self.assertGreater(dist(cube.center, bin_floor.center), 0.15)

    def test_success_criteria_are_explicit(self):
        scene = self.gen.generate(TaskSpec("把红色方块放进盒子"), self.robot, random.Random(0))
        self.assertEqual(scene.success["type"], "object_inside_container")
        self.assertEqual(scene.success["object"], "cube")


if __name__ == "__main__":
    unittest.main()
