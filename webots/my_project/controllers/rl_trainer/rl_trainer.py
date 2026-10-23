"""
Webots controller that trains the RL (PPO) agent inside the simulation.

Setup:
  - my_world.wbt sets the E-puck's controller to "rl_trainer" with supervisor TRUE,
    and has the red cube (DEF GOAL).
  - Webots must use the project's Python so torch is available:
    Tools > Preferences > Python command -> <project>/.venv/bin/python

Weights are saved like run_training.py does (resume, checkpoints, Ctrl+C safe).
Use fast mode in Webots (>> button) to train quicker.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "Train_RL"))

from run_training import display_results, execute_training_loop
import train_rl
from webots_env import WebotsDrivingEnv

ITERATIONS = 200
LEARNING_RATE = 3e-4
MODEL_FILENAME = "rl_model.pt"

save_path = ROOT_DIR / "temp_models" / MODEL_FILENAME
checkpoint_path = ROOT_DIR / "temp_modelparts" / f"part_{MODEL_FILENAME}"

model, step_fn, method_name = train_rl.setup(lr=LEARNING_RATE, env=WebotsDrivingEnv())
loop_stats = execute_training_loop(model, step_fn, ITERATIONS, checkpoint_path, save_path)

display_results({
    "name": f"{method_name} [Webots]{' [Resumed]' if loop_stats['resumed'] else ''}",
    "iterations": loop_stats["completed_epochs"],
    "start_loss": loop_stats["start_loss"],
    "final_loss": loop_stats["final_loss"],
    "save_path": str(save_path),
    "checkpoint_path": str(checkpoint_path),
})
