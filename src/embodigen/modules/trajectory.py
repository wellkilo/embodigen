"""TrajectoryGen: scene -> a successful trajectory.

Waypoint interpolation with smoothstep easing. The morphology only changes the
action space mapping and the number of controlled joints; the planning logic is
shared, which is the point of the abstraction.
"""

from __future__ import annotations

import math

from embodigen.geometry import add, dist, lerp, mul, smoothstep, sub
from embodigen.types import Frame, SceneSpec, Trajectory, Vec3

GRASP_OFFSET = 0.021


class WaypointTrajectoryGen:
    name = "waypoint-v0"

    def __init__(self, fps: float = 10.0, seconds_per_segment: float = 0.55) -> None:
        self.fps = fps
        self.seconds_per_segment = seconds_per_segment

    def plan(self, scene: SceneSpec, robot: dict) -> Trajectory:
        cube = scene.object("cube")
        bin_floor = scene.object("bin_floor")
        action_dim = int(robot.get("action_dim", 7))
        gripper_open = float(robot.get("gripper_open", 0.04))

        # Start inside the camera frustum so every frame of the episode is
        # trackable; a home pose outside the view would leave untracked frames.
        start: Vec3 = (0.02, 0.0, 0.24)
        above_cube = add(cube.center, (0.0, 0.0, 0.12))
        at_cube = add(cube.center, (0.0, 0.0, GRASP_OFFSET))
        lifted = add(at_cube, (0.0, 0.0, 0.14))
        above_bin = add(bin_floor.center, (0.0, 0.0, 0.16))
        at_bin = add(bin_floor.center, (0.0, 0.0, 0.055))
        retreat = add(at_bin, (0.0, 0.0, 0.16))

        waypoints = [
            ("home", start, gripper_open, 0.30),
            ("pre_grasp", above_cube, gripper_open, 0.35),
            ("grasp", at_cube, gripper_open, 0.45),
            ("close", at_cube, 0.0, 0.40),
            ("lift", lifted, 0.0, 0.45),
            ("transport", above_bin, 0.0, 0.55),
            ("place", at_bin, 0.0, 0.45),
            ("release", at_bin, gripper_open, 0.35),
            ("retreat", retreat, gripper_open, 0.40),
        ]

        frames: list[Frame] = []
        t = 0.0
        prev_eef = start
        prev_gripper = gripper_open
        cube_pos = cube.center
        attached = False
        resting_z = bin_floor.center[2] + bin_floor.half_extent[2] + cube.half_extent[2]
        resting: Vec3 | None = None

        for index in range(1, len(waypoints)):
            name, target, gripper, seconds = waypoints[index]
            n = max(1, int(round(seconds * self.fps)))
            if name == "retreat":
                # Fingers are open: the cube stays where it was released.
                attached = False
                if resting is not None:
                    cube_pos = resting
            for k in range(1, n + 1):
                s = smoothstep(k / n)
                eef = lerp(prev_eef, target, s)
                g = prev_gripper + (gripper - prev_gripper) * s
                if name in ("lift", "transport", "place", "release"):
                    attached = True
                if attached:
                    cube_pos = add(eef, (0.0, 0.0, -GRASP_OFFSET))
                elif resting is not None:
                    cube_pos = resting
                dt = 1.0 / self.fps
                t += dt
                delta = sub(eef, prev_eef)
                action = self._action(delta, dt, g, action_dim, robot)
                frames.append(
                    Frame(
                        t=round(t, 4),
                        eef=eef,
                        gripper=round(g, 4),
                        action=action,
                        joints=self._joints(eef, g, robot),
                        objects={"cube": cube_pos},
                        phase=name,
                    )
                )
            prev_eef = target
            prev_gripper = gripper
            if name == "release":
                # The cube settles on the bin floor once the fingers open.
                resting = (cube_pos[0], cube_pos[1], resting_z)
                cube_pos = resting

        return Trajectory(
            fps=self.fps,
            action_dim=action_dim,
            frames=frames,
            waypoints=[
                {"name": name, "eef": list(target), "gripper": gripper}
                for name, target, gripper, _ in waypoints
            ],
        )

    @staticmethod
    def _action(
        delta: Vec3, dt: float, gripper: float, action_dim: int, robot: dict
    ) -> tuple[float, ...]:
        vel = mul(delta, 1.0 / dt)
        morphology = robot.get("morphology", "arm")
        if morphology == "mobile":
            base_delta = sub(mul(delta, 1.0), (0.0, 0.0, delta[2]))
            yaw = math.atan2(base_delta[1], base_delta[0] + 1e-9)
            values = [base_delta[0] / dt, base_delta[1] / dt, yaw, gripper]
        elif morphology == "humanoid":
            values = [vel[0], vel[1], vel[2], 0.0, 0.0, 0.0, gripper]
        else:
            values = [vel[0], vel[1], vel[2], 0.0, 0.0, 0.0, gripper]
        values = values[:action_dim]
        while len(values) < action_dim:
            values.append(0.0)
        return tuple(round(v, 6) for v in values)

    @staticmethod
    def _joints(eef: Vec3, gripper: float, robot: dict) -> tuple[float, ...]:
        """Placeholder kinematics: a documented stand-in for the real solver."""
        dof = int(robot.get("dof", 7))
        joints = [
            round(math.atan2(eef[1], eef[0] + 1e-9), 5),
            round(math.atan2(eef[2], math.hypot(eef[0], eef[1]) + 1e-9), 5),
        ]
        while len(joints) < dof:
            joints.append(0.0)
        joints.append(round(gripper, 5))
        return tuple(joints)


def path_length(points: list[Vec3]) -> float:
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))
