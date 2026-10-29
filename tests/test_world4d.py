"""4D Gaussian-splat worlds: formats, exact geometry, and the lift from engine depth."""

import json
import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from embodigen.cli import main as cli
from embodigen.geometry import CameraModel
from embodigen.gameplay.pipeline import run_rollouts
from embodigen.gameplay.registry import WorldRegistry
from embodigen.modules.pack import read_episode
from embodigen.pipeline import Pipeline, PipelineConfig
from embodigen.world4d.game import recording_world
from embodigen.world4d.gaussians import SH_C0, Gaussians, quat_from_z, read_ply
from embodigen.world4d.robot import episode_world
from embodigen.world4d.world import FORMAT, write_world


def rotate(q, v):
    w, x, y, z = q
    # v' = q v q*, written out
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx))


class FormatTest(unittest.TestCase):
    def test_surfel_quaternion_turns_z_onto_the_normal(self):
        for n in [(1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0.6, 0, 0.8), (-0.48, 0.6, -0.64)]:
            got = rotate(quat_from_z(n), (0.0, 0.0, 1.0))
            self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(got, n)), (n, got))

    def test_ply_and_splat_round_trip(self):
        g = Gaussians()
        g.surfel((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.01, (0.2, 0.4, 0.8), alpha=0.9)
        g.surfel((-1.0, 0.5, 0.0), (1.0, 0.0, 0.0), 0.02, (1.0, 0.0, 0.0))
        with tempfile.TemporaryDirectory() as tmp:
            cols = read_ply(g.write_ply(Path(tmp) / "a.ply"))
            self.assertEqual(len(cols["x"]), 2)
            self.assertAlmostEqual(cols["f_dc_2"][0] * SH_C0 + 0.5, 0.8, places=5)
            self.assertAlmostEqual(1 / (1 + math.exp(-cols["opacity"][0])), 0.9, places=5)
            self.assertAlmostEqual(math.exp(cols["scale_0"][1]), 0.02, places=6)
            self.assertAlmostEqual(math.exp(cols["scale_2"][1]), 0.02 * 0.12, places=6)
            blob = g.splat_bytes()
        self.assertEqual(len(blob), 64)
        x, y, z, sx, sy, sz, r, gg, b, a, qw, qx, qy, qz = struct.unpack("<6f8B", blob[32:64])
        self.assertEqual((x, y, z, r, a), (-1.0, 0.5, 0.0, 255, 242))
        self.assertAlmostEqual(sx, 0.02, places=6)


class RobotWorldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cfg = PipelineConfig(out_dir=cls.tmp / "out", resolution=48, inject_drift_mm=0.0)
        cls.episode = Path(Pipeline(cfg).run("把红色方块放进盒子").root)
        cls.world = episode_world(cls.episode, spacing=0.008)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_moved_objects_follow_the_trajectory(self):
        scene, frames, _ = read_episode(self.episode)
        cube = next(o for o in scene.objects if o.name in frames[0].objects)
        self.assertEqual(self.world.frames, len(frames))
        for i in (0, len(frames) // 2, len(frames) - 1):
            pos = self.world.dynamic[i].pos
            xs, ys = pos[0::3], pos[1::3]
            target = frames[i].objects[cube.name]
            # the cube's splats sit within its half extent of where the episode says it is
            near = [k for k in range(len(xs)) if abs(xs[k] - target[0]) <= cube.half_extent[0] + 1e-6
                    and abs(ys[k] - target[1]) <= cube.half_extent[1] + 1e-6]
            self.assertGreater(len(near), 20)

    def test_splats_cover_what_the_camera_rendered(self):
        scene, frames, _ = read_episode(self.episode)
        cam = CameraModel(scene.camera)
        w, h = scene.camera.width, scene.camera.height
        for i in (0, len(frames) - 1):
            covered = set()
            for g in (self.world.static, self.world.dynamic[i]):
                for k in range(len(g)):
                    u, v, z = cam.project(tuple(g.pos[3 * k : 3 * k + 3]))
                    if z > 0:
                        for du in (0, 1):
                            for dv in (0, 1):
                                covered.add((int(u + du), int(v + dv)))
            objects = [(p % w, p // w) for p, s in enumerate(frames[i].seg) if s >= 2]  # everything but the table
            hit = sum((u, v) in covered for u, v in objects)
            self.assertGreater(hit / len(objects), 0.97, f"frame {i}: {hit}/{len(objects)} object pixels covered")

    def test_world_files(self):
        out = self.tmp / "world"
        r = write_world(self.world, out, ply="first")
        meta = json.loads((out / "world.json").read_text())
        self.assertEqual(meta["format"], FORMAT)
        self.assertEqual(meta["frames"], self.world.frames)
        self.assertEqual((out / "static.splat").stat().st_size, 32 * meta["static"]["count"])
        f0 = meta["per_frame"][0]
        self.assertEqual((out / f0["dynamic"]).stat().st_size, 32 * f0["count"])
        self.assertEqual(len(read_ply(out / f0["ply"])["x"]), meta["static"]["count"] + f0["count"])
        self.assertEqual(r["ply"], 1)
        self.assertEqual(len(meta["cameras"]), self.world.frames)
        with self.assertRaises(SystemExit):  # never clobbers a directory that is not a world
            (self.tmp / "other").mkdir()
            write_world(self.world, self.tmp / "other")


class GameWorldTest(unittest.TestCase):
    def test_lift_puts_walls_on_the_grid_and_floor_at_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = WorldRegistry.load()
            rows = run_rollouts(registry, ["raycast-arena"], 1, 3.0, Path(tmp), size=(48, 30), gt=True)
            world = recording_world(Path(tmp) / rows[0]["recording_id"], voxel=0.08, registry=registry)
            self.assertEqual(world.frames, 15)  # 3 s at 5 Hz
            g = world.static
            on_grid = floor = 0
            for k in range(len(g)):
                x, y, z = g.pos[3 * k : 3 * k + 3]
                if abs(z) < 1e-4:
                    floor += 1
                elif min(abs(x - round(x)), abs(y - round(y))) < 0.02 and -0.02 <= z <= 1.02:
                    on_grid += 1
            self.assertGreater(floor, 100)
            self.assertGreater((floor + on_grid) / len(g), 0.97)
            high_off_grid = [k for k in range(len(g)) if g.pos[3 * k + 2] > 0.98
                             and min(abs(g.pos[3 * k] - round(g.pos[3 * k])), abs(g.pos[3 * k + 1] - round(g.pos[3 * k + 1]))) > 0.02]
            self.assertEqual(high_off_grid, [], "ceiling is left out by default (only wall tops reach z = 1)")
            self.assertTrue(any(len(d) for d in world.dynamic), "entities should show up in some frame")
            cam = world.views[0]
            self.assertAlmostEqual(cam.eye[2], 0.5)

    def test_dataset_recordings_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "r"
            raw.mkdir()
            (raw / "recording.json").write_text(json.dumps({
                "recording_id": "r", "world": "minecraft", "source": "vpt", "stream": "s", "duration": 1.0,
                "frame_hz": 20.0, "width": 640, "height": 360}))
            with self.assertRaises(SystemExit):
                recording_world(raw)


class CliTest(unittest.TestCase):
    def test_prompt_to_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli(["world", "--task", "把红色方块放进盒子", "--resolution", "48", "--work", f"{tmp}/work",
                        "--out", f"{tmp}/w", "--ply", "none", "--spacing", "0.01"])
            self.assertEqual(code, 0)
            meta = json.loads((Path(tmp) / "w" / "world.json").read_text())
            self.assertEqual(meta["source"]["instruction"], "把红色方块放进盒子")
            self.assertTrue(meta["source"]["passed"])
            with self.assertRaises(SystemExit):
                cli(["world", "--task", "x", "--rollout", "raycast-arena"])


if __name__ == "__main__":
    unittest.main()
