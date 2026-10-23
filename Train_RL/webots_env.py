"""
Webots version of the driving environment. Only importable inside a Webots
controller process (it needs Webots' `controller` module).

The world (webots/my_project/worlds/my_world.wbt) needs:
  - the car (DEF CAR) with `supervisor TRUE`, so the controller can move things on reset
  - a Solid with DEF GOAL: the red cube to drive into
  - Solids DEF OBSTACLE_0 ... OBSTACLE_<N-1>, each with a Box as its first child's geometry

Like the real car, the only sensor is the 64x64 camera. Collisions and reaching the
goal are detected by the supervisor (contact points and positions), not by sensors.
Every reset randomizes the car, the goal and all obstacles (position, rotation, size, color).
"""

import math
import random

import torch
from PIL import Image

from arkitektur import preprocess_image
from environments import DrivingEnv, compute_reward

MAX_WHEEL_SPEED = 15.0        # rad/s, must match maxVelocity of the wheel motors in my_world.wbt
STEPS_PER_ACTION = 4          # Simulation timesteps per agent decision (4 x 16 ms)
MAX_EPISODE_STEPS = 600       # ~38 s of driving, the arena is bigger than in SimulatedCubeEnv

# --- Arena (meters, centered on the world origin) ---
ARENA_HALF_SIZE = 1.39        # 2.78 x 2.78 m inside the walls
NUM_OBSTACLES = 10
OBSTACLE_BASE_RANGE = (0.11, 0.19)    # Side of the square base, average 15 cm
OBSTACLE_HEIGHT_RANGE = (0.20, 0.30)  # Average 25 cm
OBSTACLE_COLORS = [                    # Never red, so only the goal looks like the goal
    (0.85, 0.85, 0.82), (0.55, 0.42, 0.28), (0.30, 0.32, 0.35),
    (0.20, 0.35, 0.60), (0.15, 0.15, 0.15), (0.65, 0.60, 0.50),
]
OBSTACLE_GAP = 0.30           # Free space between obstacles, so the 16.4 cm wide car can always get through
OBSTACLE_WALL_GAP = 0.20      # Free space between obstacles and walls

# --- Goal: red 6 x 6 x 8 cm cube ---
GOAL_HALF_DIAGONAL = 0.043
GOAL_Z = 0.04
GOAL_CLEARANCE = 0.15         # Free space around the goal so it can be driven into
GOAL_REACH_DISTANCE = 0.07    # Car nose to goal center that counts as reached (cube half side is 0.03)
MIN_START_DISTANCE = 0.6      # Car nose to goal at the start of an episode

# --- Car (origin on the floor under the wheel axle, +X forward) ---
CAR_NOSE_OFFSET = 0.151       # Front of the car, in front of the axle
CAR_CENTER_OFFSET = 0.056     # Middle of the footprint, in front of the axle
CAR_RADIUS = 0.13             # Circle around the footprint (19 x 16.4 cm)
CAR_CLEARANCE = 0.05          # Free space around the car at the start
CAR_Z = 0.0
CAR_PITCH = math.atan2(0.009, 0.104)  # Nose-down tilt: the caster is 9 mm shorter than the wheels, 10.4 cm in front of them
CONTACT_HEIGHT = 0.01         # Contact points higher than this are not the wheels/caster on the floor

MAX_LAYOUT_ATTEMPTS = 200


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

        self.robot_node = self.robot.getSelf()
        self.goal_node = self.robot.getFromDef("GOAL")
        if self.robot_node is None or self.goal_node is None:
            raise RuntimeError("Robot needs 'supervisor TRUE' and the world needs a Solid with DEF GOAL.")
        self.robot_node.enableContactPointsTracking(self.timestep, True)

        self.obstacles = []
        for i in range(NUM_OBSTACLES):
            node = self.robot.getFromDef(f"OBSTACLE_{i}")
            if node is None:
                raise RuntimeError(f"The world needs {NUM_OBSTACLES} Solids named DEF OBSTACLE_0 ... OBSTACLE_{NUM_OBSTACLES - 1}.")
            shape = node.getField("children").getMFNode(0)
            self.obstacles.append({
                "translation": node.getField("translation"),
                "rotation": node.getField("rotation"),
                "size": shape.getField("geometry").getSFNode().getField("size"),
                "color": shape.getField("appearance").getSFNode().getField("baseColor"),
            })

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

    def _nose_position(self) -> tuple[float, float]:
        x, y, _ = self.robot_node.getPosition()
        forward = self.robot_node.getOrientation()  # 3x3 rotation matrix, row-major
        return x + forward[0] * CAR_NOSE_OFFSET, y + forward[3] * CAR_NOSE_OFFSET

    def _distance(self) -> float:
        """Car nose to goal center, on the floor plane."""
        goal_pos = self.goal_node.getPosition()
        return math.dist(self._nose_position(), goal_pos[:2])

    def _contacts(self) -> tuple[bool, bool]:
        """(touched the goal, hit something else). Floor contacts of wheels and caster are ignored."""
        goal_xy = self.goal_node.getPosition()[:2]
        touched_goal = hit_other = False
        for contact in self.robot_node.getContactPoints(True):
            if contact.point[2] < CONTACT_HEIGHT:
                continue
            if math.dist(contact.point[:2], goal_xy) < GOAL_HALF_DIAGONAL + 0.01:
                touched_goal = True
            else:
                hit_other = True
        return touched_goal, hit_other

    # --- Random layout ---
    def _random_point(self, margin: float) -> tuple[float, float]:
        limit = ARENA_HALF_SIZE - margin
        return self.rng.uniform(-limit, limit), self.rng.uniform(-limit, limit)

    @staticmethod
    def _is_free(point, radius: float, circles, clearance: float) -> bool:
        return all(math.dist(point, center) >= radius + r + clearance for center, r in circles)

    def _random_layout(self):
        """Obstacles, goal and car placed so nothing overlaps and every obstacle can be driven around."""
        for _ in range(MAX_LAYOUT_ATTEMPTS):
            obstacles = []  # (center, base, height, yaw)
            circles = []    # (center, radius) of everything placed so far
            for _ in range(NUM_OBSTACLES):
                base = self.rng.uniform(*OBSTACLE_BASE_RANGE)
                radius = base / math.sqrt(2)
                for _ in range(MAX_LAYOUT_ATTEMPTS):
                    center = self._random_point(radius + OBSTACLE_WALL_GAP)
                    if self._is_free(center, radius, circles, OBSTACLE_GAP):
                        break
                else:
                    break
                obstacles.append((center, base, self.rng.uniform(*OBSTACLE_HEIGHT_RANGE), self.rng.uniform(0, math.pi / 2)))
                circles.append((center, radius))
            if len(obstacles) < NUM_OBSTACLES:
                continue

            for _ in range(MAX_LAYOUT_ATTEMPTS):
                goal = self._random_point(GOAL_HALF_DIAGONAL + GOAL_CLEARANCE)
                if self._is_free(goal, GOAL_HALF_DIAGONAL, circles, GOAL_CLEARANCE):
                    break
            else:
                continue

            for _ in range(MAX_LAYOUT_ATTEMPTS):
                center = self._random_point(CAR_RADIUS + CAR_CLEARANCE)
                yaw = self.rng.uniform(-math.pi, math.pi)
                heading = (math.cos(yaw), math.sin(yaw))
                nose = (center[0] + heading[0] * (CAR_NOSE_OFFSET - CAR_CENTER_OFFSET),
                        center[1] + heading[1] * (CAR_NOSE_OFFSET - CAR_CENTER_OFFSET))
                if (self._is_free(center, CAR_RADIUS, circles + [(goal, GOAL_HALF_DIAGONAL)], CAR_CLEARANCE)
                        and math.dist(nose, goal) >= MIN_START_DISTANCE):
                    car = (center[0] - heading[0] * CAR_CENTER_OFFSET, center[1] - heading[1] * CAR_CENTER_OFFSET, yaw)
                    return obstacles, goal, car
        raise RuntimeError("Could not find a random layout, loosen the spacing constants in webots_env.py.")

    @staticmethod
    def _tilted_rotation(yaw: float) -> list[float]:
        """Axis-angle for yaw around Z followed by CAR_PITCH, so the car spawns resting on its caster."""
        cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
        cy, sy = math.cos(CAR_PITCH / 2), math.sin(CAR_PITCH / 2)
        axis = (-sz * sy, cz * sy, cy * sz)
        norm = math.hypot(*axis)
        return [axis[0] / norm, axis[1] / norm, axis[2] / norm, 2 * math.acos(cz * cy)]

    def reset(self) -> torch.Tensor:
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        obstacles, goal, (car_x, car_y, car_yaw) = self._random_layout()
        for fields, (center, base, height, yaw) in zip(self.obstacles, obstacles):
            fields["size"].setSFVec3f([base, base, height])
            fields["translation"].setSFVec3f([center[0], center[1], height / 2])
            fields["rotation"].setSFRotation([0, 0, 1, yaw])
            fields["color"].setSFColor(list(self.rng.choice(OBSTACLE_COLORS)))
        self.goal_node.getField("translation").setSFVec3f([goal[0], goal[1], GOAL_Z])
        self.goal_node.getField("rotation").setSFRotation([0, 0, 1, self.rng.uniform(0, math.pi / 2)])
        self.robot_node.getField("translation").setSFVec3f([car_x, car_y, CAR_Z])
        self.robot_node.getField("rotation").setSFRotation(self._tilted_rotation(car_yaw))
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
        touched_goal, hit_other = self._contacts()
        reached = touched_goal or distance < GOAL_REACH_DISTANCE
        collided = not reached and hit_other

        reward = compute_reward(self.prev_distance, distance, reached, collided)
        self.prev_distance = distance
        done = reached or collided or self.steps >= MAX_EPISODE_STEPS

        return self._observation(), reward, done, {"reached": reached, "collided": collided}
