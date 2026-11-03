"""Optional physics tests: actual contacts, released containment, and Gym contracts."""

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodigen.gym import ENV_ID, available, make, register_envs
from embodigen.gym.demo import ScriptedPolicy
from embodigen.gym.worker import Session
from embodigen.real2sim.twin import Twin, write_twin
from embodigen.ui import server as ui


class OptionalGymTest(unittest.TestCase):
    def test_importing_core_does_not_load_the_optional_backend(self):
        code = ("import sys; import embodigen.cli; import embodigen.gym; "
                "assert all(n not in sys.modules for n in ('gymnasium','mujoco','numpy')); "
                "from unittest.mock import patch; "
                "p=patch('embodigen.gym.available',return_value=False); p.start(); "
                "\ntry: embodigen.gym.make()\n"
                "except RuntimeError as e: assert \".[gym]\" in str(e)\n"
                "else: raise AssertionError('missing extra must fail clearly')\n")
        subprocess.run([sys.executable, "-c", code], check=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True)


@unittest.skipUnless(available(), "install .[gym] for MuJoCo / Gymnasium tests")
class PhysicsGymTest(unittest.TestCase):
    def env(self, **kwargs):
        env = make(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_gymnasium_registration_checker_and_seed_reproducibility(self):
        import gymnasium
        import numpy as np
        from gymnasium.utils.env_checker import check_env

        register_envs()
        env = gymnasium.make(ENV_ID).unwrapped
        self.addCleanup(env.close)
        check_env(env, skip_render_check=True)
        first, info = env.reset(seed=12)
        original = env.snapshot()
        action = [.2, -.3, .1, 1]
        result = env.step(action)
        again, again_info = env.reset(seed=12)
        self.assertEqual(info, again_info)
        for k in first:
            np.testing.assert_array_equal(first[k], again[k])
        repeated = env.step(action)
        for k in result[0]:
            np.testing.assert_array_equal(result[0][k], repeated[0][k])
        self.assertEqual(result[1:], repeated[1:])
        env.reset(seed=13)
        self.assertNotEqual(original["cube"], env.snapshot()["cube"])
        self.assertNotEqual(info["parameters"], env.snapshot()["info"]["parameters"])

    def test_actions_are_causal_and_bad_actions_cannot_mutate_state(self):
        env = self.env(randomize=False)
        env.reset(seed=0)
        before = env.snapshot()
        for action in ([0, 1], [0, 0, 0, float("nan")], [2, 0, 0, 0], [0, 0, 0, float("inf")], None):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(before, env.snapshot())
        env.step([1, 0, 0, 1])
        after = env.snapshot()
        self.assertGreater(after["eef"][0], before["eef"][0] + .005)
        self.assertLess(math.dist(after["cube"], before["cube"]), .001)
        self.assertFalse(after["info"]["is_success"])
        self.assertNotIn("equality", env.model_xml)  # no welded grasp or hidden attachment

    def test_real_grasp_lift_drop_and_success_over_randomised_scenes(self):
        env = self.env()
        for seed in range(20):
            with self.subTest(seed=seed):
                env.reset(seed=seed)
                policy = ScriptedPolicy()
                grasped = lifted = False
                for _ in range(env.max_steps):
                    action = policy.action(env.snapshot())
                    _, reward, terminated, truncated, info = env.step(action)
                    grasped |= info["grasped"]
                    lifted |= env.cube[2] > .12
                    if info["grasped"]:
                        self.assertFalse(info["is_success"])
                    if terminated or truncated:
                        break
                self.assertTrue(grasped, "must lift with real finger contact")
                self.assertTrue(lifted)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["is_success"], info)
                self.assertGreater(env.opening, .045)
                self.assertAlmostEqual(env.cube[2], env.goal[2], delta=.004)
                with self.assertRaises(RuntimeError):
                    env.step([0, 0, 0, 1])

    def test_held_cube_above_the_bin_and_partial_overlap_are_not_success(self):
        import mujoco

        env = self.env(randomize=False)
        env.reset(seed=0)
        policy = ScriptedPolicy()
        while policy.phase < 6:
            env.step(policy.action(env.snapshot()))
        for _ in range(10):
            _, _, terminated, _, info = env.step([0, 0, 0, -1])
            self.assertFalse(terminated)
            self.assertFalse(info["is_success"])
        # Put a static test pose at the bin edge: centre inside, cube partly outside.
        addr = int(env.model.jnt_qposadr[env.model.joint("cube_free").id])
        env.data.qpos[addr:addr + 3] = [env.goal[0] + .035, env.goal[1], env.goal[2]]
        env.data.qvel[:] = 0
        env.data.qpos[env._finger_q] = .045
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._success_candidate())

    def test_time_limit_is_truncation_and_reset_clears_the_episode(self):
        env = self.env(max_steps=2, reward_mode="sparse")
        env.reset(seed=0)
        self.assertEqual(env.step([0, 0, 0, 1])[1:4], (0.0, False, False))
        _, reward, terminated, truncated, info = env.step([0, 0, 0, 1])
        self.assertEqual((reward, terminated, truncated), (0.0, False, True))
        self.assertEqual(info["reason"], "time_limit")
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0, 1])
        env.reset(seed=0)
        self.assertEqual(env.steps, 0)
        self.assertFalse(env.snapshot()["done"])

    def test_twins_keep_their_source_and_bad_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for half in (.015, .03):
                path = write_twin(Twin((-.07, .08), (.14, -.06), cube_half=half,
                                       source={"kind": "photo-vlm", "estimated": True}), Path(tmp) / "twin.json")
                env = self.env(twin=path, jitter=.03)
                _, info = env.reset(seed=2)
                self.assertTrue(info["twin_source"]["estimated"])
                self.assertTrue(any(n.startswith("twin=") for n in env.scene.notes))
                self.assertAlmostEqual(env.scene.object("cube").half_extent[0], half)
                policy = ScriptedPolicy()
                for _ in range(env.max_steps):
                    _, _, terminated, truncated, info = env.step(policy.action(env.snapshot()))
                    if terminated or truncated:
                        break
                self.assertTrue(info["is_success"], (half, info))
        for cfg in ({"max_steps": 0}, {"resolution": 2000}, {"jitter": .02},
                    {"jitter": float("nan")}, {"observation_mode": "bad"}):
            with self.assertRaises(ValueError):
                make(**cfg)

    def test_session_records_transition_pairs_and_refuses_stale_episodes(self):
        s = Session()
        self.addCleanup(s.close)
        state = s.request({"op": "reset", "config": {"seed": 0, "max_steps": 2}})
        initial = state
        first = s.request({"op": "step", "episode": state["episode"], "action": [1, 0, 0, 1]})
        rec = s.request({"op": "recording", "episode": state["episode"]})
        self.assertEqual(rec["initial_state"], initial)
        self.assertEqual(rec["transitions"][0]["state"], first)
        self.assertEqual(rec["transitions"][0]["observation"], rec["initial_observation"])
        self.assertNotEqual(rec["transitions"][0]["observation"], rec["transitions"][0]["next_observation"])
        self.assertIn("not fidelity-certified", rec["certification"])
        with self.assertRaises(ValueError):
            s.request({"op": "reset", "config": {"max_steps": -1}})
        self.assertEqual(s.state()["episode"], state["episode"])
        new = s.request({"op": "reset", "config": {"seed": 1}})
        with self.assertRaises(ValueError):
            s.request({"op": "step", "episode": state["episode"], "action": [0, 0, 0, 1]})
        self.assertEqual(s.state()["episode"], new["episode"])
        json.dumps(rec, allow_nan=False)


class GymHTTPTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = ui.make_server(Path(self.tmp.name), port=0, token="gym-test", quiet=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def call(self, path, body=None, token="gym-test"):
        req = urllib.request.Request(self.base + path, headers={"X-Token": token})
        if body is not None:
            req.data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_gym_routes_need_auth_and_keep_twins_inside_the_workspace(self):
        self.assertEqual(self.call("/api/gym/reset", {}, token="")[0], 401)
        self.assertIsNone(self.server.gym.proc)
        self.assertEqual(self.call("/api/gym/reset", {"twin": "../outside.json"})[0], 400)
        self.assertIsNone(self.server.gym.proc)

    @unittest.skipUnless(available(), "install .[gym] for worker integration")
    def test_http_worker_reset_step_export_and_shutdown(self):
        code, state = self.call("/api/gym/reset", {"seed": 5, "max_steps": 2})
        self.assertEqual(code, 200, state)
        proc = self.server.gym.proc
        code, first = self.call("/api/gym/step", {"episode": state["episode"], "action": [1, 0, 0, 1]})
        self.assertEqual(code, 200, first)
        self.assertEqual(first["steps"], 1)
        self.assertEqual(self.call("/api/gym/step", {"episode": "stale", "action": [0, 0, 0, 1]})[0], 400)
        self.assertEqual(self.call("/api/gym/step", {"episode": state["episode"], "action": [2, 0, 0, 1]})[0], 400)
        _, current = self.call("/api/gym/state")
        self.assertEqual(current["steps"], 1)
        _, final = self.call("/api/gym/step", {"episode": state["episode"], "action": [0, 0, 0, 1]})
        self.assertTrue(final["truncated"])
        _, rec = self.call("/api/gym/recording", {"episode": state["episode"]})
        self.assertEqual(len(rec["transitions"]), 2)
        self.assertEqual(rec["transitions"][0]["next_observation"], rec["transitions"][1]["observation"])
        self.server.gym.close()
        self.assertIsNotNone(proc.poll(), "console shutdown must not leave a physics worker running")


if __name__ == "__main__":
    unittest.main()
