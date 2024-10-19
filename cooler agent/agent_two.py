import time
import json
from env import Gym2OpEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

env = Gym2OpEnv()
gym_env = DummyVecEnv(
    [
        lambda: Monitor(
            env._gym_env
        )
    ]
)

model = SAC(
    "MlpPolicy",
    gym_env,
    learning_rate=0.0001,  # Adjusted learning rate for stability
    batch_size=256,        # Increased batch size for better updates
    tau=0.005,
    gamma=0.90,            # Adjusted discount factor for longer-term rewards
    target_update_interval=10,  # More frequent target updates
    ent_coef="auto",       # Keep auto, but monitor its decay
    verbose=1,
    tensorboard_log="./tb_logs/",
    device="cuda",
    buffer_size=10000,
    learning_starts=2000,
)

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./models",
    log_path="./logs",
    deterministic=True,
    eval_freq=5000,
)

callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=10000, tb_log_name="SAC_TRAIN" + str(time.time()), **kwargs
)

# Save policy weights
model.save("PPO_GRID.pt")


# Load policy weights
model.load("PPO_GRID.pt")
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
