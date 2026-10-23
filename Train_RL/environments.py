"""
Environments and reward system for Deep Reinforcement Learning.

Every environment follows the same small interface (a Markov Decision Process):
    state = env.reset()
    state, reward, done, info = env.step((left_wheel, right_wheel))

state is a camera frame as a float tensor (3, 64, 64) in [0, 1], exactly what
DrivingNetwork gets in BC and SemiSL. Distances are in meters.

SimulatedCubeEnv is a tiny 2D stand-in for Webots, so the PPO code can be
tested on any computer. WebotsDrivingEnv (webots_env.py) is the real one.
"""

import math
import random

import torch

# --- Reward system (shared by all environments) ---
REWARD_GOAL = 1.0         # Reached the red cube
REWARD_COLLISION = -1.0   # Drove into a wall
PROGRESS_SCALE = 2.0      # Reward per meter driven closer to the cube (punishes driving away)
TIME_PENALTY = 0.002      # Small cost per step, so dawdling isn't free

GOAL_RADIUS = 0.08          # Center-to-center distance that counts as "reached"
MAX_EPISODE_STEPS = 300

# --- Robot for SimulatedCubeEnv (a small e-puck sized stand-in, the Webots car is in webots_env.py) ---
MAX_STEP_DISTANCE = 0.0165  # Meters driven per step at full speed
AXLE_WIDTH = 0.052
ROBOT_RADIUS = 0.037
CUBE_HALF_SIZE = 0.025


def compute_reward(prev_distance: float, distance: float, reached: bool, collided: bool) -> float:
    """Reward for one step: + for reaching the cube, - for crashing, shaping for getting closer."""
    if reached:
        return REWARD_GOAL
    if collided:
        return REWARD_COLLISION
    return PROGRESS_SCALE * (prev_distance - distance) - TIME_PENALTY


class DrivingEnv:
    """Interface every environment implements."""
    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action) -> tuple[torch.Tensor, float, bool, dict]:
        raise NotImplementedError


class SimulatedCubeEnv(DrivingEnv):
    """Flat square arena with one red cube, rendered as a very simple 64x64 camera image."""
    IMAGE_SIZE = 64
    FIELD_OF_VIEW = 0.84  # Radians, same as the e-puck camera
    ARENA_HALF_SIZE = 0.5

    def __init__(self, start_visible: bool = True, seed: int | None = None):
        self.start_visible = start_visible  # Start with the cube in view (easier task)
        self.rng = random.Random(seed)
        self.focal = (self.IMAGE_SIZE / 2) / math.tan(self.FIELD_OF_VIEW / 2)

    def _distance(self) -> float:
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def reset(self) -> torch.Tensor:
        limit = self.ARENA_HALF_SIZE - 0.1
        while True:
            self.x, self.y = self.rng.uniform(-limit, limit), self.rng.uniform(-limit, limit)
            self.goal_x, self.goal_y = self.rng.uniform(-limit, limit), self.rng.uniform(-limit, limit)
            if self._distance() > 0.25:
                break

        angle_to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        spread = self.FIELD_OF_VIEW / 2 * 0.8 if self.start_visible else math.pi
        self.heading = angle_to_goal + self.rng.uniform(-spread, spread)

        self.steps = 0
        self.prev_distance = self._distance()
        return self._render()

    def step(self, action):
        left = max(-1.0, min(1.0, float(action[0])))
        right = max(-1.0, min(1.0, float(action[1])))

        # Differential drive: average speed moves forward, speed difference turns
        forward = (left + right) / 2 * MAX_STEP_DISTANCE
        turn = (right - left) * MAX_STEP_DISTANCE / AXLE_WIDTH
        self.heading += turn
        self.x += forward * math.cos(self.heading)
        self.y += forward * math.sin(self.heading)
        self.steps += 1

        distance = self._distance()
        wall = self.ARENA_HALF_SIZE - ROBOT_RADIUS
        collided = abs(self.x) > wall or abs(self.y) > wall
        reached = distance < GOAL_RADIUS

        reward = compute_reward(self.prev_distance, distance, reached, collided)
        self.prev_distance = distance
        done = reached or collided or self.steps >= MAX_EPISODE_STEPS

        return self._render(), reward, done, {"reached": reached, "collided": collided}

    def _render(self) -> torch.Tensor:
        size = self.IMAGE_SIZE
        image = torch.empty(3, size, size)
        image[:, : size // 2] = 0.75  # Walls
        image[:, size // 2 :] = 0.45  # Floor
        image += torch.randn_like(image) * 0.02

        # Cube position in the robot's view: forward distance and sideways offset
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        ahead = dx * math.cos(self.heading) + dy * math.sin(self.heading)
        right = dx * math.sin(self.heading) - dy * math.cos(self.heading)

        if ahead > 0.03:
            angle = math.atan2(right, ahead)
            if abs(angle) < self.FIELD_OF_VIEW / 2 + 0.1:
                center = size / 2 + math.tan(angle) * self.focal
                half = max(1.0, CUBE_HALF_SIZE * self.focal / ahead)
                x0, x1 = int(max(0, center - half)), int(min(size, center + half))
                y0, y1 = int(max(0, size / 2 - half / 2)), int(min(size, size / 2 + half * 1.5))
                if x1 > x0 and y1 > y0:
                    image[0, y0:y1, x0:x1] = 0.85
                    image[1:, y0:y1, x0:x1] = 0.1

        return image.clamp(0.0, 1.0)
