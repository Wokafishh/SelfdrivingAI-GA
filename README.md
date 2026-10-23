# SelfdrivingAI-GA
Python code for self driving models. Three different models will be trained with different algorithmns.

Använder:
- PyTorch
- Webots (endast för RL-träningen)

## Setup (.venv)
`.venv` is not in git and can't be copied between Linux and Windows, so build it on each computer.

Linux / no NVIDIA GPU (CPU training):
```
python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
```

Windows + NVIDIA GPU (e.g. RTX 3060, CUDA training). Needs an up-to-date NVIDIA driver:
```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-cuda.txt
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line should print `True` and the GPU name.

## Training
`python run_training.py` asks for everything, including the device (cpu/cuda).
Or pass it all at once, e.g. a 5 minute session on the GPU (run again to resume):
```
python run_training.py --method bc --minutes 5 --data a --name bc_final --device cuda
```
Models are always saved as CPU tensors, so a model trained on the GPU works in `server/ModelServer.py` on any computer.

Several intervals in a row, e.g. 6 x 5 minutes unattended:
```
python run_training.py --method bc --minutes 5 --intervals 6 --data a --name bc_final --device cuda
```
After each interval the model is saved as `temp_models/bc_final_session1.pt` ... `_session5.pt`, and after the last one as
`bc_final_final.pt`. The iterations per interval are printed in a table at the end and logged to
`training_logs/bc_final_intervals.csv`. `bc_final_best.pt` is always the model with the lowest validation loss.

`--method rl` in `run_training.py` only trains in `SimulatedCubeEnv`, a simple 2D stand-in for testing the PPO code. Real RL training runs inside Webots, see below.

## RL training in Webots (Windows + NVIDIA GPU)
The arena is `webots/my_project/worlds/my_world.wbt`: 2.78 x 2.78 m floor, 0.68 m walls, 10 obstacles and a red 6x6x8 cm goal cube.
Our car (meshes in `Train_RL/models/webots`) has a 64x64 camera as its only sensor. Every episode the car, goal and
obstacles are placed randomly (`Train_RL/webots_env.py`). The training loop is `webots/my_project/controllers/rl_trainer/rl_trainer.py`.

1. Install Webots R2025a and build the `.venv` with `requirements-cuda.txt` (see Setup).
2. In Webots: Tools > Preferences > Python command -> the full path to `<project>\.venv\Scripts\python.exe`
   (otherwise the controller can't import torch).
3. Open `webots/my_project/worlds/my_world.wbt` and press fast mode (>>). Training starts on its own and uses
   the GPU if one is found (`DEVICE = "auto"`).
4. A session stops after `TIME_LIMIT_MINUTES` (5 by default) and saves `temp_models/rl_final.pt` plus a
   `_session<N>` snapshot. Reload the world (or reset the simulation) to run another session; it resumes from the last one.
   Logs go to `training_logs/rl_final_*.csv`.

Session length and model name can also be set without editing code, with the `RL_MINUTES` and `RL_MODEL_NAME`
environment variables. Set them in the same terminal before starting Webots:
```
set RL_MINUTES=120
"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" webots\my_project\worlds\my_world.wbt
```
Headless on Linux: `PATH="$PWD/.venv/bin:$PATH" RL_MINUTES=120 webots --batch --mode=fast --no-rendering webots/my_project/worlds/my_world.wbt`

Tips:
- Learning from scratch is slow (one PPO iteration is ~1 minute). Starting from a BC model helps a lot: set `INIT_WEIGHTS`
  in `Train_RL/train_rl.py` to e.g. `ROOT_DIR / "temp_models" / "bc20.pt"`. `.pt` files are not in git, so copy that file over by hand.
- Car settings in the world file that are estimates and should match the real car: wheel `maxVelocity` (also
  `MAX_WHEEL_SPEED` in `webots_env.py`), camera `fieldOfView` and the camera position/angle.
- If the Inventor model changes, re-export it to `Train_RL/models` and run `python Train_RL/models/convert_car_model.py`.


### Behaviour cloning/Imitation Learning
Inputs: Labeled data: Expert looks at state S (probably a frame of a video) and picks action A (driving forward or r/l). 
Training: The model is given a state S and tries to predict the expert's action A. It is given supervised loss based on how close it is to A. criterion() + loss.backward() in pytorch.

### Deep Reinforcement Learning / Proximal Policy Optimization
Inputs: A (randomized?) webots environment, and an algorithm for determining reward.
Training: The model is continuously given states S and has to decide action Â. Actions compound in 3d space and after a certain in-world time is up the model is evaluated on how close it has made it to an arbitrary goal. It receives reward for this. Reward can be given for other arbitrary actions, such as turning right. In this way we can demotivate destructive behaviours.
We will use the WeBots + webots api for training.

### Self-Supervised / Semi-Supervised Learning
Inputs: A small batch of labeled data (state S and expert action A) and a large batch of unlabeled data (state S only).
Training: A Student model learns via gradients while a Teacher model tracks the Student as a smoothed moving average. Supervised loss compares Student predicts on clean state S to expert action A. Consistency loss compares Student predictions on noisy state S to Teacher predictions on clean state S. The Student updates through standard backpropagation, and the Teacher updates by blending its weights with the Student's new weights.
