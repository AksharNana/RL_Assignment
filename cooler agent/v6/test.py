from dqn import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import Gym2OpEnv
import json
import time

# Load the environment
env = Gym2OpEnv()
gym_env = DummyVecEnv([lambda: Monitor(env)])

# Load the best model
model_path = "./models/best_model"  # Path where the best model is saved
model = DQN.load(model_path, gym_env)

# Testing parameters
nb_episode_test = 5
seeds_test_env = (0, 1, 2, 3, 4, 5)    # Same size as nb_episode_test
seeds_test_agent = (2, 3, 4, 5, 6, 7)  # Same size as nb_episode_test
ts_ep_test =  (0, 1, 2, 3, 4, 5)       # Same size as nb_episode_test

ep_infos = {}  # Information that will be saved

total_cum_reward = 0
total_steps_survived = 0

# Loop through each test episode
for ep_test_num in range(nb_episode_test):
    init_obs, init_infos = env.reset(seed=seeds_test_env[ep_test_num])
    model.set_random_seed(seeds_test_agent[ep_test_num])
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
        "time serie id": ts_ep_test[ep_test_num],
        "time serie folder": env._gym_env.init_env.chronics_handler.get_id(),
        "env seed": seeds_test_env[ep_test_num],
        "agent seed": seeds_test_agent[ep_test_num],
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
