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

