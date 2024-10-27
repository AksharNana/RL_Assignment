from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import Gym2OpEnv
import json
import time

env = Gym2OpEnv()
gym_env = DummyVecEnv([lambda: Monitor(env)])

policy_kwargs = dict(net_arch=[128, 128])  # Two layers of 128 units each

# DQN model
model = DQN(
    "MlpPolicy",
    gym_env,
    learning_rate=0.001,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log="./tb_logs/",
     policy_kwargs=policy_kwargs  # Use the custom architecture
)

# Train the model
model.learn(total_timesteps=15000)

# Save and load the model
model.save("DQN_GRID.pt")
model.load("DQN_GRID.pt")

nb_episode_test = 5
seeds_test_env = (0, 1, 2, 3, 4, 5)    # same size as nb_episode_test
seeds_test_agent = (3, 4, 5, 6, 7)  # same size as nb_episode_test
ts_ep_test =  (0, 1, 2, 3, 4, 5)       # same size as nb_episode_test

ep_infos = {}  # information that will be saved

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
    ep_infos[ep_test_num] = {"time serie id": ts_ep_test[ep_test_num],
                             "time serie folder": env._gym_env.init_env.chronics_handler.get_id(),
                             "env seed": seeds_test_env[ep_test_num],
                             "agent seed": seeds_test_agent[ep_test_num],
                             "steps survived": step_survived,
                             "total steps": int(env._gym_env.init_env.max_episode_duration()),
                             "cum reward": cum_reward}
    
print(json.dumps(ep_infos, indent=4))


'''
RESULTS

{
    "0": {
        "time serie id": 0,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0117",
        "env seed": 0,
        "agent seed": 3,
        "steps survived": 1092,
        "total steps": 8064,
        "cum reward": 263.58903414011
    },
    "1": {
        "time serie id": 1,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0118",
        "env seed": 1,
        "agent seed": 4,
        "steps survived": 2493,
        "total steps": 8064,
        "cum reward": 596.2483571693301
    },
    "2": {
        "time serie id": 2,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0119",
        "env seed": 2,
        "agent seed": 5,
        "steps survived": 36,
        "total steps": 8064,
        "cum reward": 12.31704705953598
    },
    "3": {
        "time serie id": 3,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0120",
        "env seed": 3,
        "agent seed": 6,
        "steps survived": 510,
        "total steps": 8064,
        "cum reward": 133.0693740695715
    },
    "4": {
        "time serie id": 4,
        "time serie folder": "/home/suvarn/data_grid2op/l2rpn_case14_sandbox/chronics/0121",
        "env seed": 4,
        "agent seed": 7,
        "steps survived": 751,
        "total steps": 8064,
        "cum reward": 177.98008447885513
    }
}
'''