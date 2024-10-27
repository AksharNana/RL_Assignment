from dqn import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from replay_buffer import PrioritizedReplayBuffer
from early_stopping import EarlyStoppingCallback
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from env import Gym2OpEnv
import json
import time
import numpy as np



# Environment setup
env = Gym2OpEnv()
gym_env = DummyVecEnv([lambda: Monitor(env)])

# Define policy architecture
policy_kwargs = dict(net_arch=[128, 128])  

# Model setup
model = DQN(
    "MlpPolicy",
    gym_env,
    learning_rate=0.001,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.02,
    target_update_interval=250,
    buffer_size=100000,
    verbose=1,
    tensorboard_log="./tb_logs/",
    policy_kwargs = policy_kwargs,
    replay_buffer_class=PrioritizedReplayBuffer
)

# Evaluation setup
eval_callback = EvalCallback(
    env,
    n_eval_episodes=5,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,
)

# Early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    eval_freq=500000,
    patience=50,
    min_timesteps=200000,  
    verbose=1,
    best_model_save_path="./models_rollout",

)



# Train model with both callbacks
model.learn(
    total_timesteps=100000,
    tb_log_name="DQN_TRAIN_" + str(time.time()),
    callback=[eval_callback, early_stopping_callback]
)

# Save the final model
model.save("DQN_FINAL_GRID.pt")
# Load the best model
model_path = "./models/best_model"  # Path where the best model is saved
model = DQN.load(model_path, gym_env)
# Testing the agent
nb_episode_test = 5
seeds_test_env = (0, 1, 2, 3, 4, 5)    # Same size as nb_episode_test
seeds_test_agent = (2, 3, 4, 5, 6, 7)  # Same size as nb_episode_test
ts_ep_test =  (0, 1, 2, 3, 4, 5)       # Same size as nb_episode_test

ep_infos = {}  # Information that will be saved

total_cum_reward = 0
total_steps_survived = 0

for ep_test_num in range(nb_episode_test):
    init_obs, init_infos = env.reset(seed=seeds_test_env[ep_test_num])
    model.set_random_seed(seeds_test_agent[ep_test_num])
    done = False
    cum_reward = 0
    step_survived = 0
    obs = init_obs
    while not done:
        act, _states = model.predict(obs, deterministic=True)
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

print(json.dumps(ep_infos, indent=4))


avg_rew = total_cum_reward / 10
avg_step = total_steps_survived / 10
print("###########")
print("# SUMMARY #")
print("###########")
print(f"Average reward = {avg_rew}")
print(f"total steps survived= {avg_step}")
print("###########")



'''
RESULTS

Step 55000, mean reward: 109.69
Stopping training at step 55000 due to no improvement.
{
    "0": {
        "time serie id": 0,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0264",
        "env seed": 0,
        "agent seed": 2,
        "steps survived": 2818,
        "total steps": 8064,
        "cum reward": 990.8136456012726
    },
    "1": {
        "time serie id": 1,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0265",
        "env seed": 1,
        "agent seed": 3,
        "steps survived": 2822,
        "total steps": 8064,
        "cum reward": 957.5585479736328
    },
    "2": {
        "time serie id": 2,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0266",
        "env seed": 2,
        "agent seed": 4,
        "steps survived": 6452,
        "total steps": 8064,
        "cum reward": 2258.7024085223675
    },
    "3": {
        "time serie id": 3,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0267",
        "env seed": 3,
        "agent seed": 5,
        "steps survived": 1091,
        "total steps": 8064,
        "cum reward": 383.7996412217617
    },
    "4": {
        "time serie id": 4,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0268",
        "env seed": 4,
        "agent seed": 6,
        "steps survived": 5001,
        "total steps": 8064,
        "cum reward": 1725.6886424273252
    }
}
###########
# SUMMARY #
###########
Average reward = 631.6562885746359
total steps survived= 1818.4
###########

'''