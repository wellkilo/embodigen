"""CLI-owned live Gym session. JSON-lines transport keeps physics out of the web console."""

from __future__ import annotations

import json
import secrets
import sys
from dataclasses import asdict

from embodigen.gym import ENV_ID, make
from embodigen.gym.demo import ScriptedPolicy


class Session:
    def __init__(self):
        self.env = None
        self.policy = ScriptedPolicy()
        self.episode = ""
        self.recording = None

    def request(self, command: dict):
        op = command.get("op")
        if op == "reset":
            cfg = dict(command.get("config") or {})
            seed = cfg.pop("seed", 0)
            # Build and reset first: a bad request never destroys the current episode.
            env = make(**cfg)
            try:
                obs, info = env.reset(seed=seed)
            except BaseException:
                env.close()
                raise
            if self.env:
                self.env.close()
            self.env, self.policy = env, ScriptedPolicy()
            self.episode = secrets.token_hex(8)
            self.obs = {k: v.tolist() for k, v in obs.items()}
            initial = self.state()
            self.recording = {"format": "embodigen-gym-rollout-v1", "env_id": ENV_ID,
                              "backend": "mujoco", "robot": "cartesian-parallel-gripper",
                              "physics_version": env.physics_version,
                              "episode": self.episode, "seed": seed, "config": cfg,
                              "parameters": dict(env.parameters), "scene": asdict(env.scene),
                              "model_xml": env.model_xml,
                              "initial_state": initial, "initial_observation": self.obs, "transitions": [],
                              "certification": "interactive physics rollout; not fidelity-certified"}
            return initial
        if self.env is None:
            raise ValueError("请先重置场景")
        if op == "state":
            return self.state()
        if command.get("episode") != self.episode:
            raise ValueError("场景已重置，请刷新当前回合后再操作")
        if op == "step":
            demo = command.get("demo", False)
            if not isinstance(demo, bool):
                raise ValueError("demo must be a boolean")
            action = self.policy.action(self.env.snapshot()) if demo else command.get("action")
            before = self.obs
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs = {k: v.tolist() for k, v in obs.items()}
            state = self.state()
            state.update(reward=reward, terminated=terminated, truncated=truncated,
                         action=list(action), controller="scripted-demo" if demo else "external")
            self.recording["transitions"].append({"observation": before, "action": list(action),
                                                 "next_observation": self.obs, "reward": reward,
                                                 "terminated": terminated, "truncated": truncated,
                                                 "state": state})
            return state
        if op == "recording":
            return self.recording
        raise ValueError(f"unknown Gym operation: {op}")

    def state(self):
        return {**self.env.snapshot(), "episode": self.episode}

    def close(self):
        if self.env:
            self.env.close()


def run() -> int:
    session = Session()
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError("command must be an object")
                response = {"result": session.request(command)}
            except (ValueError, TypeError, RuntimeError, OSError, SystemExit) as exc:
                response = {"error": str(exc)}
            print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        session.close()
    return 0
