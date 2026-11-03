"""Feedback-based scripted smoke-test policy, not a trained model."""

from __future__ import annotations

import math


class ScriptedPolicy:
    def __init__(self):
        self.phase, self.ticks = 0, 0

    def action(self, state: dict) -> list[float]:
        cube, eef, goal = state["cube"], state["eef"], state["goal"]
        grasp_z = max(.021, cube[2])  # small cubes sit below the gripper's 20-mm Z limit
        targets = [[cube[0], cube[1], .17], [cube[0], cube[1], grasp_z],
                   [cube[0], cube[1], grasp_z], [eef[0], eef[1], .17],
                   [goal[0], goal[1], .17], [goal[0], goal[1], .11],
                   [goal[0], goal[1], .11], [goal[0], goal[1], .17]]
        target = targets[min(self.phase, 7)]
        self.ticks += 1
        if self.phase in (2, 6):
            ready = self.ticks >= (20 if self.phase == 2 else 25)
        else:
            ready = math.dist(eef, target) < .004
        grip = -1.0 if 2 <= self.phase <= 5 else 1.0
        action = [max(-1.0, min(1.0, (target[i] - eef[i]) / .015)) for i in range(3)] + [grip]
        if ready and self.phase < 7:
            self.phase += 1
            self.ticks = 0
        return action
