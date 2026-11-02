"""Real -> sim: a measured or photo-estimated twin becomes certified, randomised training data."""

import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from embodigen.cli import main as cli
from embodigen.gameplay.llm import ChatClient, LLMConfig, LLMError
from embodigen.pipeline import Pipeline, PipelineConfig
from embodigen.real2sim.estimate import TwinEstimator, parse_estimate, twin_from_estimate
from embodigen.real2sim.twin import Twin, TwinSceneGenerator, load_twin, problems, write_twin
from embodigen.robots import load_robot
from embodigen.types import Camera, TaskSpec

ROBOT = load_robot("franka")
TASK = TaskSpec(instruction="put the cube in the bin", morphology="arm", robot="franka", seed=0)
GOOD = Twin(cube_xy=(-0.07, 0.08), bin_xy=(0.14, -0.06), cube_rgb=(66, 102, 176))


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            code = cli(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
            out.write(str(exc.code) if not isinstance(exc.code, int) else "")
    return code, out.getvalue()


class TwinTest(unittest.TestCase):
    def test_file_round_trip_and_stable_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_twin(GOOD, Path(tmp) / "t" / "twin.json")
            back = load_twin(path)
        self.assertEqual(back.to_json(), GOOD.to_json())
        self.assertEqual(back.twin_id, GOOD.twin_id)
        other = Twin(cube_xy=(-0.07, 0.09), bin_xy=(0.14, -0.06))
        self.assertNotEqual(other.twin_id, GOOD.twin_id)
        with self.assertRaises(SystemExit):
            Twin.from_json({"format": "something-else"})

    def test_what_the_planner_cannot_run_is_named(self):
        self.assertEqual(problems(GOOD, ROBOT, Camera().at_resolution(64)), [])
        far = Twin(cube_xy=(0.9, 0.0), bin_xy=(0.14, -0.06))
        self.assertTrue(any("beyond" in p for p in problems(far, ROBOT)))
        self.assertTrue(any("outside the camera view" in p for p in problems(far, ROBOT)))
        touching = Twin(cube_xy=(0.10, -0.06), bin_xy=(0.14, -0.06))
        self.assertTrue(any("overlaps the bin" in p for p in problems(touching, ROBOT)))
        huge = Twin(cube_xy=(-0.07, 0.08), bin_xy=(0.14, -0.06), cube_half=0.2)
        self.assertTrue(any("half-size" in p for p in problems(huge, ROBOT)))

    def test_jitter_zero_is_the_twin_exactly(self):
        scene = TwinSceneGenerator(Camera().at_resolution(64), GOOD).generate(TASK, ROBOT, random.Random(0))
        cube = scene.object("cube")
        self.assertEqual((cube.center[0], cube.center[1], cube.albedo), (-0.07, 0.08, (66, 102, 176)))
        self.assertEqual(scene.object("bin_floor").center[:2], (0.14, -0.06))
        self.assertIn(f"twin={GOOD.twin_id}", scene.notes)
        self.assertIn("twin-source=measured by hand", scene.notes)

    def test_jitter_makes_bounded_reachable_variants(self):
        gen = TwinSceneGenerator(Camera().at_resolution(64), GOOD, jitter=0.03)
        seen = set()
        for seed in range(10):
            scene = gen.generate(TASK, ROBOT, random.Random(seed))
            cube = scene.object("cube")
            self.assertLessEqual(abs(cube.center[0] - GOOD.cube_xy[0]), 0.03 + 1e-9)
            self.assertLessEqual(abs(cube.center[1] - GOOD.cube_xy[1]), 0.03 + 1e-9)
            twin = Twin(cube_xy=cube.center[:2], bin_xy=scene.object("bin_floor").center[:2])
            self.assertEqual(problems(twin, ROBOT, scene.camera), [])
            self.assertLessEqual(max(abs(c - o) for c, o in zip(cube.albedo, GOOD.cube_rgb)), 18)
            seen.add((round(cube.center[0], 4), round(cube.center[1], 4)))
        self.assertGreater(len(seen), 8)  # actually different scenes
        again = gen.generate(TASK, ROBOT, random.Random(3)).object("cube").center
        self.assertEqual(again, gen.generate(TASK, ROBOT, random.Random(3)).object("cube").center)

    def test_an_unrunnable_twin_is_refused_not_bent(self):
        far = Twin(cube_xy=(0.9, 0.0), bin_xy=(0.14, -0.06))
        with self.assertRaises(SystemExit) as ctx:
            TwinSceneGenerator(Camera().at_resolution(64), far).generate(TASK, ROBOT, random.Random(0))
        self.assertIn("cannot be run", str(ctx.exception))


class PipelineTest(unittest.TestCase):
    def test_twin_episodes_pass_the_same_fidelity_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            twin = write_twin(GOOD, Path(tmp) / "twin.json")
            cfg = PipelineConfig(out_dir=Path(tmp) / "out", resolution=48, twin=str(twin), jitter=0.03)
            rows = Pipeline(cfg).run_batch("put the cube in the bin", "franka", count=3)
            self.assertTrue(all(r["passed"] for r in rows), rows)
            self.assertEqual({r["twin"] for r in rows}, {GOOD.twin_id})
            ep = Path(rows[0]["path"])
            notes = json.loads((ep / "meta" / "scene.json").read_text())["notes"]
            self.assertIn(f"twin={GOOD.twin_id}", notes)
            index = json.loads((Path(tmp) / "out" / "index.json").read_text())
            self.assertEqual({e["twin"] for e in index["episodes"]}, {GOOD.twin_id})
            plain = PipelineConfig(out_dir=Path(tmp) / "x", resolution=48)
            self.assertNotEqual(cfg.fingerprint(), plain.fingerprint())
            self.assertNotEqual(cfg.fingerprint(), PipelineConfig(out_dir=Path(tmp), resolution=48, twin=str(twin),
                                                                   jitter=0.0).fingerprint())

    def test_template_scenes_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = Pipeline(PipelineConfig(out_dir=Path(tmp), resolution=48)).run_batch("把红色方块放进盒子", "franka")[0]
        self.assertNotIn("twin", row)


class CliTest(unittest.TestCase):
    def test_measured_twin_then_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/twin.json"
            code, text = run_cli("real2sim", "--cube-xy", "-0.07", "0.08", "--bin-xy", "0.14", "-0.06",
                                 "--cube-color", "blue", "--out", out)
            self.assertEqual(code, 0, text)
            self.assertIn("measured by hand", text)
            self.assertEqual(load_twin(out).cube_rgb, (66, 102, 176))
            code, text = run_cli("gen", "--task", "put the cube in the bin", "--twin", out, "--jitter", "0.02",
                                 "-n", "2", "-j", "1", "--resolution", "48", "--out", f"{tmp}/o")
            self.assertEqual(code, 0, text)
            self.assertIn("certified    2/2", text)

    def test_unreachable_measurement_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = run_cli("real2sim", "--cube-xy", "0.9", "0", "--bin-xy", "0.1", "0", "--out", f"{tmp}/t.json")
            self.assertEqual(code, 1)
            self.assertIn("beyond", text)
            self.assertFalse((Path(tmp) / "t.json").exists())

    def test_modes_and_jitter_need_their_inputs(self):
        for argv in (["real2sim"], ["real2sim", "--image", "a.png", "--cube-xy", "0", "0", "--bin-xy", "0.1", "0"],
                     ["real2sim", "--cube-xy", "0", "0"],
                     ["gen", "--task", "x", "--jitter", "0.02"]):
            code, _ = run_cli(*argv)
            self.assertNotEqual(code, 0, argv)


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def reply(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


GOOD_ESTIMATE = {"cube_xy": [-0.06, 0.09], "cube_edge_cm": 4.5, "cube_rgb": [60, 100, 180],
                 "bin_xy": [0.15, -0.05], "bin_rgb": [90, 94, 100], "table_rgb": [190, 185, 170], "confidence": 0.6}
KEY = "sk-test-not-a-real-key"


class EstimateTest(unittest.TestCase):
    def cfg(self, key=KEY):
        return LLMConfig(provider="custom", model="vlm-test", base_url="http://127.0.0.1:9/v1", api_key=key)

    def photo(self, tmp):
        path = Path(tmp) / "scene.jpg"
        path.write_bytes(b"\xff\xd8\xff not really a jpeg")
        return path

    def test_reply_becomes_a_labelled_estimate_and_the_key_stays_out(self):
        seen = []

        def opener(request, timeout):
            seen.append(json.loads(request.data))
            return FakeResponse(reply("Here you go:\n```json\n" + json.dumps(GOOD_ESTIMATE) + "\n```"))

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg()
            twin = TwinEstimator(cfg, ChatClient(cfg, opener=opener)).estimate(self.photo(tmp))
            path = write_twin(twin, Path(tmp) / "twin.json")
            written = path.read_text()
        self.assertTrue(twin.source["estimated"])
        self.assertEqual((twin.source["kind"], twin.source["model"], twin.source["confidence"]), ("photo-vlm", "vlm-test", 0.6))
        self.assertAlmostEqual(twin.cube_half, 0.0225)
        self.assertNotIn(KEY, written)
        self.assertNotIn(KEY, json.dumps(seen))  # the key is a header, never in the body
        image = seen[0]["messages"][1]["content"][0]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/jpeg;base64,"))
        self.assertEqual(problems(twin, ROBOT), [])
        scene = TwinSceneGenerator(Camera().at_resolution(64), twin).generate(TASK, ROBOT, random.Random(0))
        self.assertIn("twin-source=model-estimated from a photo", scene.notes)

    def test_no_key_means_no_request_and_no_photo_read(self):
        def opener(request, timeout):
            raise AssertionError("a request went out without a key")

        cfg = self.cfg(key="")
        with self.assertRaises(SystemExit) as ctx:
            TwinEstimator(cfg, ChatClient(cfg, opener=opener)).estimate(Path("/no/such/photo.jpg"))
        self.assertIn("API key", str(ctx.exception))

    def test_bad_replies_are_errors_and_wild_numbers_are_clamped(self):
        with self.assertRaises(LLMError):
            parse_estimate("I cannot see a cube")
        with self.assertRaises(LLMError):
            twin_from_estimate({"cube_xy": [0, 0]}, {})
        wild = twin_from_estimate({**GOOD_ESTIMATE, "cube_xy": [5.0, -9.0], "cube_edge_cm": 80, "cube_rgb": [300, -5, 7]}, {})
        self.assertEqual(wild.cube_xy, (0.35, -0.35))
        self.assertEqual((wild.cube_half, wild.cube_rgb), (0.03, (255, 0, 7)))
        self.assertTrue(problems(wild, ROBOT))  # clamped, then still judged by the planner's limits

    def test_cli_estimate_uses_the_users_provider_settings(self):
        env = {"EMBODIGEN_CONFIG_DIR": tempfile.mkdtemp(), "EMBODIGEN_LLM_PROVIDER": "custom",
               "EMBODIGEN_LLM_BASE_URL": "http://127.0.0.1:9/v1", "EMBODIGEN_LLM_MODEL": "vlm-test",
               "EMBODIGEN_LLM_API_KEY": KEY}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env), \
                mock.patch.object(ChatClient, "complete", return_value=reply(json.dumps(GOOD_ESTIMATE))):
            code, text = run_cli("real2sim", "--image", str(self.photo(tmp)), "--out", f"{tmp}/t.json")
            self.assertEqual(code, 0, text)
            self.assertIn("model-estimated from a photo", text)
            self.assertIn("confidence   0.6", text)
            self.assertNotIn(KEY, text)
            self.assertNotIn(KEY, (Path(tmp) / "t.json").read_text())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {**env, "EMBODIGEN_LLM_API_KEY": ""}):
            code, text = run_cli("real2sim", "--image", str(self.photo(tmp)), "--out", f"{tmp}/t.json")
            self.assertNotEqual(code, 0)
            self.assertFalse((Path(tmp) / "t.json").exists())


if __name__ == "__main__":
    unittest.main()
