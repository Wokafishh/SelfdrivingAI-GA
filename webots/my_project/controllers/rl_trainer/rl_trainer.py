"""
Webots controller that trains the RL (PPO) agent inside the simulation.

Setup:
  - my_world.wbt has our car (DEF CAR, meshes from Train_RL/models/webots) with
    controller "rl_trainer" and supervisor TRUE, the red cube (DEF GOAL) and 10 obstacles
    (DEF OBSTACLE_0..9). WebotsDrivingEnv re-randomizes all of them every episode.
  - Webots must use the project's Python so torch is available:
    Tools > Preferences > Python command -> <project>/.venv/bin/python

Uses run_training.run_session: same resuming, time-limited sessions, snapshots and CSV logs.
Use fast mode in Webots (>> button) to train quicker.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "Train_RL"))

from run_training import display_results, run_session
from training_common import describe_device, resolve_device
import train_rl
from webots_env import WebotsDrivingEnv

# RL_MINUTES / RL_MODEL_NAME environment variables override these, e.g. for a short test run
TIME_LIMIT_MINUTES = float(os.environ.get("RL_MINUTES", 5))  # Wall-clock training time per session
ITERATIONS = None        # Max iterations per session (None = until the time runs out)
LEARNING_RATE = 3e-4
DEVICE = "auto"          # Webots has no keyboard prompt: auto = cuda if available, else cpu
MODEL_FILENAME = os.environ.get("RL_MODEL_NAME", "rl_final.pt")  # Same name every session = resume; snapshots get _session<N>

device = resolve_device(DEVICE)
print(f"[DEVICE] Training on {describe_device(device)}")
model, step_fn, method_name = train_rl.setup(lr=LEARNING_RATE, env=WebotsDrivingEnv(), device=device)
results = run_session(
    model, step_fn, f"{method_name} [Webots]",
    save_path=ROOT_DIR / "temp_models" / MODEL_FILENAME,
    checkpoint_path=ROOT_DIR / "temp_modelparts" / f"part_{MODEL_FILENAME}",
    epochs=ITERATIONS,
    time_limit_minutes=TIME_LIMIT_MINUTES,
    log_dir=ROOT_DIR / "training_logs",
    device_name=describe_device(device),
)
display_results(results)
