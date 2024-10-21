import json
import time
import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_freq, patience, min_timesteps=50000, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_timesteps = min_timesteps  # Minimum timesteps before early stopping can activate
        self.n_calls = 0
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0

    def _on_step(self) -> bool:
        self.n_calls += 1
       
        if self.n_calls < self.min_timesteps:  # Wait until minimum timesteps are reached
            return True

        if self.n_calls % self.eval_freq == 0:
            # Retrieve the training reward
            mean_reward = np.mean([info["episode"]["r"] for info in self.locals["infos"] if "episode" in info])

            if self.verbose > 0:
                print(f"Step {self.n_calls}, mean reward: {mean_reward:.2f}")

            # Check for improvement
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1

            # Stop training if no improvement for `patience` evaluations
            if self.no_improvement_steps >= self.patience:
                if self.verbose > 0:
                    print(f"Stopping training at step {self.n_calls} due to no improvement in mean reward.")
                return False

        return True
