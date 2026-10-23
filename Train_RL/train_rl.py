"""
Deep Reinforcement Learning (RL) with Proximal Policy Optimization (PPO).

The agent (DrivingNetwork) is put in an environment and acts on its own.
Each iteration:
  1. Rollout: the agent drives for ROLLOUT_STEPS steps and collects
     (state s, action a, reward r). Actions are sampled around the network's
     output, so the agent explores.
  2. Update: actions that turned out better than expected get more likely,
     worse ones less likely. PPO clips each update so the policy can't change
     too much at once.

A separate critic network estimates how much reward to expect from a state,
which tells the update what "better than expected" means.

Through run_training.py this trains in SimulatedCubeEnv (a test environment that
runs anywhere). Real training runs inside Webots via
webots/my_project/controllers/rl_trainer/rl_trainer.py.
"""

import sys
import pickle
from pathlib import Path

import torch
from torch import nn, optim
from torch.distributions import Normal

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent))
from arkitektur import DrivingNetwork
from environments import SimulatedCubeEnv

# Optionally start the actor from a BC/SemiSL model instead of random weights,
# e.g. ROOT_DIR / "temp_models" / "bc19.pt". Skips a lot of random exploration.
INIT_WEIGHTS = None

ROLLOUT_STEPS = 1024
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
GAMMA = 0.99          # Discount: how much future reward counts
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.001  # Small bonus for keeping some randomness (exploration)
INITIAL_LOG_STD = -0.5


class ActorCritic(nn.Module):
    """Actor = DrivingNetwork (same model as BC/SemiSL), critic = small separate value network."""
    def __init__(self):
        super().__init__()
        self.actor = DrivingNetwork(in_channels=3)
        self.log_std = nn.Parameter(torch.full((2,), INITIAL_LOG_STD))
        self.critic = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):
        # The actor always stays in eval mode: dropout and changing BatchNorm statistics
        # would make the action probabilities differ between rollout and update.
        super().train(mode)
        self.actor.eval()
        return self

    def distribution(self, states: torch.Tensor) -> Normal:
        left, right = self.actor(states)
        mean = torch.cat([left, right], dim=1)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(1)


def setup(lr: float = 3e-4, data_files=None, env=None):
    """data_files is unused (RL makes its own data). env defaults to SimulatedCubeEnv."""
    torch.manual_seed(42)
    env = env if env is not None else SimulatedCubeEnv(seed=42)

    policy = ActorCritic()
    if INIT_WEIGHTS is not None:
        with open(INIT_WEIGHTS, "rb") as f:
            policy.actor.load_state_dict(pickle.load(f))
        print(f"[RL] Actor initialised from '{INIT_WEIGHTS}'.")
    policy.train()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    print(f"[RL] Environment: {type(env).__name__} | {ROLLOUT_STEPS} steps per iteration")

    state = env.reset()
    episode_return, episode_length = 0.0, 0

    def collect_rollout():
        nonlocal state, episode_return, episode_length
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
        finished = []  # (return, length, reached) per finished episode

        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                dist = policy.distribution(state.unsqueeze(0))
                action = dist.sample()[0]
                log_prob = dist.log_prob(action).sum()
                value = policy.value(state.unsqueeze(0))[0]

            next_state, reward, done, info = env.step(action.clamp(-1.0, 1.0).tolist())

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))

            episode_return += reward
            episode_length += 1
            if done:
                finished.append((episode_return, episode_length, info.get("reached", False)))
                episode_return, episode_length = 0.0, 0
                state = env.reset()
            else:
                state = next_state

        with torch.no_grad():
            last_value = policy.value(state.unsqueeze(0))[0]

        # Generalized Advantage Estimation: how much better each action was than the critic expected
        # (timeouts are treated like real episode ends, a simplification)
        values_t = torch.stack(values)
        rewards_t = torch.tensor(rewards)
        dones_t = torch.tensor(dones)
        advantages = torch.zeros(ROLLOUT_STEPS)
        gae = 0.0
        for t in reversed(range(ROLLOUT_STEPS)):
            next_value = last_value if t == ROLLOUT_STEPS - 1 else values_t[t + 1]
            not_done = 1.0 - dones_t[t]
            delta = rewards_t[t] + GAMMA * next_value * not_done - values_t[t]
            gae = delta + GAMMA * GAE_LAMBDA * not_done * gae
            advantages[t] = gae

        batch = {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "advantages": advantages,
            "returns": advantages + values_t,
        }
        return batch, finished, rewards_t

    def ppo_update(batch) -> float:
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss, updates = 0.0, 0

        for _ in range(PPO_EPOCHS):
            for idx in torch.randperm(ROLLOUT_STEPS).split(MINIBATCH_SIZE):
                dist = policy.distribution(batch["states"][idx])
                log_prob = dist.log_prob(batch["actions"][idx]).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()

                ratio = torch.exp(log_prob - batch["log_probs"][idx])
                adv = advantages[idx]
                policy_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * adv,
                ).mean()
                value_loss = (policy.value(batch["states"][idx]) - batch["returns"][idx]).pow(2).mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()

                total_loss += loss.item()
                updates += 1

        return total_loss / updates

    def train_step(iteration: int) -> float:
        batch, finished, rewards = collect_rollout()
        loss = ppo_update(batch)

        message = f"[ITERATION {iteration}] PPO loss: {loss:.4f} | Reward/step: {rewards.mean():+.4f}"
        if finished:
            mean_return = sum(f[0] for f in finished) / len(finished)
            mean_length = sum(f[1] for f in finished) / len(finished)
            success = sum(f[2] for f in finished) / len(finished)
            message += (
                f" | Episodes: {len(finished)}, mean return {mean_return:+.3f}, "
                f"mean length {mean_length:.0f}, reached cube {success:.0%}"
            )
        message += f" | Action std: {policy.log_std.exp().mean():.3f}"
        print(message)
        return loss

    return policy, train_step, "Deep Reinforcement Learning (PPO)"
