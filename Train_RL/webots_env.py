"""
Webots version of the driving environment. Only importable inside a Webots
controller process (it needs Webots' `controller` module).

The world (webots/my_project/worlds/my_world.wbt) needs:
  - the robot with `supervisor TRUE`, so the controller can move things on reset
  - a Solid with DEF GOAL: the red cube to drive to
"""

import math
import random

import torch
from PIL import Image

from arkitektur import preprocess_image
from environments import (
    DrivingEnv,
    GOAL_RADIUS,
    MAX_EPISODE_STEPS,
    compute_reward,
)

MAX_WHEEL_SPEED = 6.28        # rad/s, e-puck motor limit
STEPS_PER_ACTION = 4          # Simulation timesteps per agent decision
COLLISION_THRESHOLD = 1000.0  # e-puck proximity sensor value that counts as touching a wall

ARENA_CENTER = (-0.141708, 0.092358)  # RectangleArena translation in my_world.wbt
ARENA_HALF_SIZE = 0.5                 # RectangleArena default floorSize is 1 x 1
SPAWN_MARGIN = 0.1
ROBOT_Z = 0.0
GOAL_Z = 0.025


class WebotsDrivingEnv(DrivingEnv):
    def __init__(self, seed: int | None = None):
        from controller import Supervisor

        self.rng = random.Random(seed)
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for motor in (self.left_motor, self.right_motor):
            motor.setPosition(float("inf"))  # Velocity control
            motor.setVelocity(0.0)

        self.proximity_sensors = [self.robot.getDevice(f"ps{i}") for i in range(8)]
        for sensor in self.proximity_sensors:
            sensor.enable(self.timestep)

        self.robot_node = self.robot.getSelf()
        self.goal_node = self.robot.getFromDef("GOAL")
        if self.robot_node is None or self.goal_node is None:
            raise RuntimeError("Robot needs 'supervisor TRUE' and the world needs a Solid with DEF GOAL.")

        self.steps = 0
        self.prev_distance = 0.0

    def _advance(self, timesteps: int) -> None:
        for _ in range(timesteps):
            if self.robot.step(self.timestep) == -1:
                raise SystemExit("Webots simulation ended.")

    def _observation(self) -> torch.Tensor:
        width, height = self.camera.getWidth(), self.camera.getHeight()
        image = Image.frombuffer("RGBA", (width, height), self.camera.getImage(), "raw", "BGRA", 0, 1)
        return preprocess_image(image)

    def _distance(self) -> float:
        robot_pos = self.robot_node.getPosition()
        goal_pos = self.goal_node.getPosition()
        return math.hypot(goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1])

    def _collided(self) -> bool:
        return max(sensor.getValue() for sensor in self.proximity_sensors) > COLLISION_THRESHOLD

    def _random_point(self) -> tuple[float, float]:
        limit = ARENA_HALF_SIZE - SPAWN_MARGIN
        return (
            ARENA_CENTER[0] + self.rng.uniform(-limit, limit),
            ARENA_CENTER[1] + self.rng.uniform(-limit, limit),
        )

    def reset(self) -> torch.Tensor:
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        while True:
            robot_xy, goal_xy = self._random_point(), self._random_point()
            if math.dist(robot_xy, goal_xy) > 0.25:
                break

        self.robot_node.getField("translation").setSFVec3f([robot_xy[0], robot_xy[1], ROBOT_Z])
        self.robot_node.getField("rotation").setSFRotation([0, 0, 1, self.rng.uniform(-math.pi, math.pi)])
        self.goal_node.getField("translation").setSFVec3f([goal_xy[0], goal_xy[1], GOAL_Z])
        self.robot_node.resetPhysics()
        self._advance(STEPS_PER_ACTION)

        self.steps = 0
        self.prev_distance = self._distance()
        return self._observation()

    def step(self, action):
        left = max(-1.0, min(1.0, float(action[0])))
        right = max(-1.0, min(1.0, float(action[1])))
        self.left_motor.setVelocity(left * MAX_WHEEL_SPEED)
        self.right_motor.setVelocity(right * MAX_WHEEL_SPEED)
        self._advance(STEPS_PER_ACTION)
        self.steps += 1

        distance = self._distance()
        reached = distance < GOAL_RADIUS
        collided = not reached and self._collided()

        reward = compute_reward(self.prev_distance, distance, reached, collided)
        self.prev_distance = distance
        done = reached or collided or self.steps >= MAX_EPISODE_STEPS

        return self._observation(), reward, done, {"reached": reached, "collided": collided}
