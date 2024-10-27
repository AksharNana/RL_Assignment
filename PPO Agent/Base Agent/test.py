import json
import time
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from gymnasium.wrappers.frame_stack import FrameStack

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace, ScalerAttrConverter, ContinuousToDiscreteConverter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from lightsim2grid import LightSimBackend

from early_stopping import EarlyStoppingCallback
import torch as th

class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "gen_p", "gen_q", "load_p", "load_q",
                    "actual_dispatch", "rho", "line_status", "storage_power", "storage_charge","connectivity_matrix"]

        observation_space = self._gym_env.observation_space
        self._gym_env.observation_space.close()
        obs_gym, info = self._gym_env.reset()
        observation_space = observation_space.reencode_space("gen_p",
                                   ScalerAttrConverter(substract=0.,
                                                       divide=self._g2op_env.gen_pmax
                                                       )
                                   )
        observation_space = observation_space.reencode_space("load_p",
                                        ScalerAttrConverter(substract=obs_gym["load_p"],
                                                            divide=0.5 * obs_gym["load_p"]
                                                            )
                                        )
        self._gym_env.observation_space = observation_space
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                   attr_to_keep=obs_attr_to_keep,
                                                   divide={"actual_dispatch": self._g2op_env.gen_pmax},
                                                    functs={"connectivity_matrix": (
                                                      lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                                      0., 1., None, None,
                                                    )
                                                 }
                                         )
        # export observation space for the Grid2opEnv
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high,)

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        action_space = self._gym_env.action_space
        self._gym_env.action_space.close()
        act_attr_to_keep = ["redispatch","set_storage"]
        self._gym_env.action_space = action_space
        self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space,
                                                           attr_to_keep=act_attr_to_keep)
        self.action_space = Box(shape=(self._g2op_env.action_space.n_gen,),
                                     low=self._g2op_env.action_space.gen_pmin,
                                     high=self._g2op_env.action_space.gen_pmax)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)
  
    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()

# Load the environment
env = Gym2OpEnv()
gym_env = DummyVecEnv([lambda: Monitor(env)])

# Load the best model
model_path = "best_model"  # Path where the best model is saved
model = PPO.load(model_path, gym_env)

# Testing parameters
nb_episode_test = 50
seeds_test_env = (0, 1, 2, 3, 4, 5)    # Same size as nb_episode_test
seeds_test_agent = (2, 3, 4, 5, 6, 7)  # Same size as nb_episode_test
ts_ep_test =  (0, 1, 2, 3, 4, 5)       # Same size as nb_episode_test

ep_infos = {}  # Information that will be saved

total_cum_reward = 0
total_steps_survived = 0

# Loop through each test episode
for ep_test_num in range(nb_episode_test):
    # init_obs, init_infos = env.reset(seed=seeds_test_env[ep_test_num])
    # model.set_random_seed(seeds_test_agent[ep_test_num])
    init_obs, init_infos = env.reset()
   
    done = False
    cum_reward = 0
    step_survived = 0
    obs = init_obs
    
    while not done:
        act, _states = model.predict(obs, deterministic=True)  # deterministic for testing
        obs, reward, terminated, truncated, info = env.step(act)
        step_survived += 1
        cum_reward += float(reward)
        done = terminated or truncated

    total_steps_survived += step_survived
    total_cum_reward += cum_reward

    ep_infos[ep_test_num] = {
        "time serie id": ep_test_num,
        "time serie folder": env._gym_env.init_env.chronics_handler.get_id(),
        # "env seed": seeds_test_env[ep_test_num],
        # "agent seed": seeds_test_agent[ep_test_num],
        "steps survived": step_survived,
        "total steps": int(env._gym_env.init_env.max_episode_duration()),
        "cum reward": cum_reward
    }

# Print the episode information in JSON format
print(json.dumps(ep_infos, indent=4))

# Summary
avg_rew = total_cum_reward / nb_episode_test
avg_step = total_steps_survived / nb_episode_test
print("###########")
print("# SUMMARY #")
print("###########")
print(f"Average reward = {avg_rew}")
print(f"Total steps survived= {avg_step}")
print("###########")



import matplotlib.pyplot as plt
import numpy as np

# Extract cumulative rewards and steps survived from ep_infos
episode_numbers = list(ep_infos.keys())
cumulative_rewards = [ep_infos[ep]['cum reward'] for ep in episode_numbers]
steps_survived = [ep_infos[ep]['steps survived'] for ep in episode_numbers]

# Calculate averages
average_reward = np.mean(cumulative_rewards)
average_steps = np.mean(steps_survived)

# Plot cumulative reward for each episode with a faded line, and add average line
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_numbers, cumulative_rewards, label='Cumulative Reward', marker='o', color='blue', alpha=0.3)
plt.axhline(average_reward, color='blue', linestyle='--', label=f'Average Reward = {average_reward:.2f}')
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode")
plt.grid(True)
plt.legend()

# Plot steps survived for each episode with a faded line, and add average line
plt.subplot(1, 2, 2)
plt.plot(episode_numbers, steps_survived, label='Steps Survived', marker='o', color='orange', alpha=0.3)
plt.axhline(average_steps, color='orange', linestyle='--', label=f'Average Steps = {average_steps:.2f}')
plt.xlabel("Episode")
plt.ylabel("Steps Survived")
plt.title("Steps Survived per Episode")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

