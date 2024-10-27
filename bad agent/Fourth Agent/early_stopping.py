import json
import time
import os
import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_freq, patience,best_model_save_path, min_timesteps=50000, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_timesteps = min_timesteps
        self.n_calls = 0
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0
        self.episode_rewards = []
        self.current_reward = 0  # Track cumulative reward for the current episode

    def _on_step(self) -> bool:
        self.n_calls += 1

        # Add reward for the current step to cumulative reward
        reward = self.locals["rewards"][0]  # Access the step reward
        self.current_reward += reward

        if self.locals["dones"][0]:  # Check if the episode is done
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0  # Reset cumulative reward for the next episode

        if self.n_calls < self.min_timesteps:
            return True

        if self.n_calls % self.eval_freq == 0:
            if self.episode_rewards:  # Make sure there are recorded rewards
                mean_reward = np.mean(self.episode_rewards)
                self.episode_rewards = []  # Reset episode rewards after evaluation

                if self.verbose > 0:
                    print(f"Step {self.n_calls}, mean reward: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_steps = 0
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                    print(f"New best mean reward: {self.best_mean_reward:.2f} - Saving best model.")
                    
                else:
                    self.no_improvement_steps += 1

                if self.no_improvement_steps >= self.patience:
                    if self.verbose > 0:
                        print(f"Stopping training at step {self.n_calls} due to no improvement.")
                    return False

        return True
