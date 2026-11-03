"""Real contact dynamics behind the optional [gym] extra.

This is a Cartesian parallel gripper, NOT a Franka model. Five actuated slide
joints move the palm and fingers; the cube is a free rigid body. No weld,
attachment, teleport or planned object trajectory is used during a step.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from embodigen.gym import ENV_ID
from embodigen.modules.scene import BIN_HALF, BIN_WALL, TemplateSceneGenerator
from embodigen.real2sim.twin import TwinSceneGenerator, load_twin
from embodigen.robots import load_robot
from embodigen.types import Camera, TaskSpec

DT = 0.05
SUBSTEPS = 25
MAX_DELTA = 0.015  # metres per 50 ms control step, before control randomisation
LOW = np.array([-0.38, -0.38, 0.02])
HIGH = np.array([0.38, 0.38, 0.32])


def _vec(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def _model_xml(scene, mass: float, friction: float) -> str:
    root = Element("mujoco", model="embodigen-cartesian-pick-place")
    SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast",
               cone="elliptic", iterations="80")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="512", offheight="512")
    default = SubElement(root, "default")
    SubElement(default, "geom", friction=f"{friction} 0.01 0.001", condim="4",
               solref="0.008 1", solimp="0.95 0.99 0.001")
    world = SubElement(root, "worldbody")
    SubElement(world, "light", pos="0 -0.2 1.5", dir="0 0 -1", diffuse="0.9 0.9 0.9")
    eye = np.array(scene.camera.eye)
    z = eye - np.array(scene.camera.target)
    z /= np.linalg.norm(z)
    x = np.cross(np.array(scene.camera.up), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    fovy = math.degrees(2 * math.atan(scene.camera.height / (2 * scene.camera.fy)))
    SubElement(world, "camera", name="observation", pos=_vec(eye), xyaxes=_vec([*x, *y]), fovy=str(fovy))
    for obj in scene.objects:
        parent = world
        attrs = {"name": obj.name, "type": "box", "size": _vec(obj.half_extent),
                 "rgba": _vec([*(c / 255 for c in obj.albedo), 1])}
        if obj.name == "cube":
            parent = SubElement(world, "body", name="cube", pos=_vec(obj.center))
            SubElement(parent, "freejoint", name="cube_free")
            attrs["mass"] = str(mass)
        else:
            attrs["pos"] = _vec(obj.center)
        SubElement(parent, "geom", **attrs)
    hand = SubElement(world, "body", name="hand", gravcomp="1")
    for i, axis in enumerate(("x", "y", "z")):
        unit = [0, 0, 0]
        unit[i] = 1
        SubElement(hand, "joint", name=axis, type="slide", axis=_vec(unit),
                   range=_vec([LOW[i], HIGH[i]]), damping="1")
    SubElement(hand, "geom", name="palm", type="box", pos="0 0 0.054", size="0.023 0.06 0.012",
               mass="0.4", rgba="0.35 0.42 0.5 1")
    for name, side in (("left", -1), ("right", 1)):
        finger = SubElement(hand, "body", name=name, pos=f"0 {side * .006} 0", gravcomp="1")
        SubElement(finger, "joint", name=name, type="slide", axis=f"0 {side} 0", range="0 0.045",
                   damping="0.3")
        SubElement(finger, "geom", name=f"finger_{name}", type="box", pos="0 0 0.012",
                   size="0.014 0.006 0.03", mass="0.06", rgba="0.65 0.74 0.8 1")
    actuators = SubElement(root, "actuator")
    for i, axis in enumerate(("x", "y", "z")):
        SubElement(actuators, "position", joint=axis, kp="1600", kv="80",
                   ctrlrange=_vec([LOW[i], HIGH[i]]), forcerange="-120 120")
    for name in ("left", "right"):
        SubElement(actuators, "position", joint=name, kp="250", kv="5",
                   ctrlrange="0 0.045", forcerange="-12 12")
    return tostring(root, encoding="unicode")


class PickPlaceEnv(gym.Env):
    """4 actions in [-1, 1]: dx, dy, dz, opening (-1 closed, +1 open).

    State mode needs no OpenGL. ``observation_mode='rgb'`` adds a MuJoCo RGB
    camera image to observations. ``render_mode='rgb_array'`` enables render().
    Success requires full XY containment, floor support, release, and 5 quiet
    steps. A held cube above the bin is never a success.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    physics_version = mujoco.__version__

    def __init__(self, *, twin: str | Path | None = None, jitter: float = 0.0,
                 randomize: bool = True, max_steps: int = 400, reward_mode: str = "dense",
                 observation_mode: str = "state", render_mode: str | None = None,
                 resolution: int = 128):
        if observation_mode not in ("state", "rgb") or render_mode not in (None, "rgb_array"):
            raise ValueError("observation_mode: state|rgb; render_mode: None|rgb_array")
        if reward_mode not in ("dense", "sparse"):
            raise ValueError("reward_mode must be dense or sparse")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be an integer in 1..10000")
        if not isinstance(resolution, int) or not 32 <= resolution <= 512:
            raise ValueError("resolution must be an integer in 32..512")
        if not math.isfinite(jitter) or not 0 <= jitter <= .1 or (jitter and not twin):
            raise ValueError("jitter must be in 0..0.1 and nonzero jitter requires a twin")
        self.twin = load_twin(twin) if twin else None
        self.jitter, self.randomize = jitter, bool(randomize)
        self.max_steps, self.reward_mode = max_steps, reward_mode
        self.observation_mode, self.render_mode, self.resolution = observation_mode, render_mode, resolution
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        fields = {"observation": spaces.Box(-np.inf, np.inf, (29,), np.float64),
                  "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), np.float64),
                  "desired_goal": spaces.Box(-np.inf, np.inf, (3,), np.float64)}
        if observation_mode == "rgb":
            fields["rgb"] = spaces.Box(0, 255, (resolution, resolution, 3), np.uint8)
        self.observation_space = spaces.Dict(fields)
        self.model = self.data = self.scene = self._renderer = None
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported; configure the environment at creation")
        self._close_renderer()
        # An explicit reset seed repeats the entire scene and physical parameters.
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
        rng = random.Random(self.episode_seed)
        camera = Camera().at_resolution(self.resolution)
        task = TaskSpec("把方块放进盒子", seed=self.episode_seed, template="pick_place")
        generator = (TwinSceneGenerator(camera, self.twin, self.jitter) if self.twin
                     else TemplateSceneGenerator(camera))
        self.scene = generator.generate(task, load_robot("franka"), rng)
        # The Franka config supplies the existing scene sampler's reach check only.
        for obj in (self.scene.object("cube"), self.scene.object("bin_floor")):
            if any(abs(obj.center[i]) > .29 for i in (0, 1)):
                raise ValueError("twin is outside this Gym's Cartesian workspace (centres must be within ±0.29 m)")
        if self.randomize:
            eye = tuple(e + rng.uniform(-.025, .025) for e in camera.eye)
            self.scene.camera = replace(self.scene.camera, eye=eye)
            self.scene.objects = [replace(o, albedo=tuple(max(0, min(255, c + rng.randint(-16, 16)))
                                                         for c in o.albedo)) for o in self.scene.objects]
        self.parameters = {"mass_kg": rng.uniform(.04, .08) if self.randomize else .06,
                           "friction": rng.uniform(.9, 1.3) if self.randomize else 1.1,
                           "control_gain": rng.uniform(.85, 1.15) if self.randomize else 1.0}
        xml = _model_xml(self.scene, self.parameters["mass_kg"], self.parameters["friction"])
        self.model_xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._hand = self.model.body("hand").id
        self._cube = self.model.body("cube").id
        self._cube_geom = self.model.geom("cube").id
        self._fingers = [self.model.geom("finger_" + n).id for n in ("left", "right")]
        self._hand_q = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in ("x", "y", "z")]
        self._finger_q = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in ("left", "right")]
        self.data.qpos[self._hand_q] = [0.0, 0.0, .20]
        self.data.qpos[self._finger_q] = .045
        self.data.ctrl[:] = [0, 0, .20, .045, .045]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=100)  # settle the cube onto the table
        mujoco.mj_forward(self.model, self.data)
        floor = self.scene.object("bin_floor")
        self.goal = np.array([*floor.center[:2], floor.center[2] + floor.half_extent[2]
                              + self.scene.object("cube").half_extent[2]])
        self.steps, self.return_, self._quiet = 0, 0.0, 0
        self._done = False
        self._potential = self._progress()
        return self._obs(), self._info(False, "")

    @property
    def eef(self):
        return self.data.xpos[self._hand].copy()

    @property
    def cube(self):
        return self.data.xpos[self._cube].copy()

    @property
    def opening(self) -> float:
        return float(sum(self.data.qpos[self._finger_q]))

    def _contacts(self) -> list[bool]:
        pairs = [{int(c.geom1), int(c.geom2)} for c in self.data.contact]
        return [any({self._cube_geom, f} == pair for pair in pairs) for f in self._fingers]

    def _success_candidate(self) -> bool:
        extent = np.abs(self.data.xmat[self._cube].reshape(3, 3)) @ np.array(self.scene.object("cube").half_extent)
        inside = np.all(np.abs(self.cube[:2] - self.goal[:2]) + extent[:2] < BIN_HALF - BIN_WALL)
        floor_top = self.goal[2] - self.scene.object("cube").half_extent[2]
        supported = abs(self.cube[2] - extent[2] - floor_top) < .004
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._cube, velocity, 0)
        quiet = np.linalg.norm(velocity[3:]) < .03 and np.linalg.norm(velocity[:3]) < 1.0
        released = self.opening > 2 * self.scene.object("cube").half_extent[1] + .005 and not any(self._contacts())
        return bool(inside and supported and quiet and released)

    def _progress(self) -> float:
        return float(-2 * np.linalg.norm(self.eef - self.cube) - 3 * np.linalg.norm(self.cube - self.goal))

    def step(self, action):
        if self._done:
            raise RuntimeError("episode is finished or not started; call reset() before step()")
        try:
            a = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must contain four finite numbers in [-1, 1]") from exc
        if a.shape != (4,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1.0):
            raise ValueError("action must contain four finite numbers in [-1, 1]")
        self.data.ctrl[:3] = np.clip(self.eef + a[:3] * MAX_DELTA * self.parameters["control_gain"], LOW, HIGH)
        self.data.ctrl[3:] = (a[3] + 1) * .0225
        mujoco.mj_step(self.model, self.data, nstep=SUBSTEPS)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        failed = not finite or self.cube[2] < -.05 or np.any(np.abs(self.cube[:2]) > .48)
        if not finite:
            raise RuntimeError("MuJoCo state became non-finite; reset the environment")
        self._quiet = self._quiet + 1 if self._success_candidate() and not failed else 0
        success = self._quiet >= 5
        terminated = bool(success or failed)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        collision = self._collision()
        potential = self._progress()
        reward = (10.0 * success - 2.0 * failed + potential - self._potential - .01 - .02 * collision
                  if self.reward_mode == "dense" else float(success) - float(failed))
        self._potential = potential
        self.return_ += reward
        self._done = terminated or truncated
        reason = "success" if success else "out_of_workspace" if failed else "time_limit" if truncated else ""
        return self._obs(), float(reward), terminated, truncated, self._info(success, reason)

    def _collision(self) -> bool:
        hand_geoms = {self.model.geom("palm").id, *self._fingers}
        return any((int(c.geom1) in hand_geoms or int(c.geom2) in hand_geoms)
                   and not (int(c.geom1) in hand_geoms and int(c.geom2) in hand_geoms)
                   and self._cube_geom not in (int(c.geom1), int(c.geom2))
                   for c in self.data.contact)

    def _obs(self):
        obs = {"observation": np.concatenate([self.data.qpos, self.data.qvel, self.data.ctrl, [self._quiet / 5]]),
               "achieved_goal": self.cube, "desired_goal": self.goal.copy()}
        if self.observation_mode == "rgb":
            obs["rgb"] = self._render_rgb()
        return obs

    def _info(self, success: bool, reason: str):
        return {"is_success": bool(success), "reason": reason, "grasped": all(self._contacts()),
                "collision": self._collision(), "steps": self.steps, "return": float(self.return_),
                "seed": self.episode_seed, "backend": "mujoco", "robot": "cartesian-parallel-gripper",
                "parameters": dict(self.parameters), "twin_source": dict(self.twin.source) if self.twin else None}

    def snapshot(self) -> dict:
        """JSON physics state for the browser viewer and auditable transition recordings."""
        if self.model is None:
            raise RuntimeError("call reset() first")
        geoms = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            geoms.append({"name": name, "position": self.data.geom_xpos[i].tolist(),
                          "rotation": self.data.geom_xmat[i].reshape(3, 3).tolist(),
                          "half_extent": self.model.geom_size[i].tolist(), "rgba": self.model.geom_rgba[i].tolist()})
        return {"eef": self.eef.tolist(), "cube": self.cube.tolist(), "goal": self.goal.tolist(),
                "opening": self.opening, "geoms": geoms, "camera": self.scene.camera.to_dict(),
                "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(), "ctrl": self.data.ctrl.tolist(),
                "steps": self.steps, "max_steps": self.max_steps, "return": float(self.return_),
                "done": self._done, "info": self._info(self._quiet >= 5, "success" if self._quiet >= 5 else
                                                       "time_limit" if self._done and self.steps >= self.max_steps else
                                                       "out_of_workspace" if self._done else "")}

    def _render_rgb(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.resolution, width=self.resolution)
        self._renderer.update_scene(self.data, camera="observation")
        return self._renderer.render().copy()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.model is None:
            raise RuntimeError("call reset() before render()")
        return self._render_rgb()

    def _close_renderer(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def close(self):
        self._close_renderer()
        self._done = True


def register_envs():
    if ENV_ID not in gym.envs.registry:
        gym.register(ENV_ID, entry_point="embodigen.gym.mujoco_env:PickPlaceEnv")
