import time
import json
from ray import tune
from ray.rllib.algorithms.a3c import A3CTrainer
from ray.tune.registry import register_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import Gym2OpEnv

# Initialize Ray
import ray
ray.init(ignore_reinit_error=True)

# Register the custom environment
def env_creator(env_config):
    return Gym2OpEnv()

register_env("grid2op_env", env_creator)

# Wrap the environment in a DummyVecEnv for vectorized operations (like SB3)
gym_env = DummyVecEnv(
    [
        lambda: Monitor(
            Gym2OpEnv()._gym_env
        )
    ]
)

# Define the A3C configuration similar to your SAC config
a3c_config = {
    "env": "grid2op_env",  # Your custom environment
    "num_workers": 4,  # Number of parallel workers (this is key for A3C)
    "framework": "torch",  # Use torch for GPU acceleration (can use TensorFlow if preferred)
    "lr": 0.0001,  # Adjusted learning rate for stability
    "gamma": 0.90,  # Adjusted discount factor for longer-term rewards
    "model": {
        "fcnet_hiddens": [256, 256],  # Similar to the SAC network configuration
        "fcnet_activation": "relu",  # Activation function
    },
    "rollout_fragment_length": 20,  # Number of steps before computing a gradient update
    "train_batch_size": 256,  # Similar to your SAC batch size
    "use_gae": True,  # Use Generalized Advantage Estimation (optional)
    "lambda": 0.95,  # GAE lambda parameter
    "num_sgd_iter": 10,  # Number of iterations for stochastic gradient descent
}

# Use Ray's A3C trainer to train the model
trainer = A3CTrainer(config=a3c_config)

# Callbacks, similar to SB3
callbacks = []
eval_callback = EvalCallback(
    gym_env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./models",
    log_path="./logs",
    deterministic=True,
    eval_freq=5000,
)
callbacks.append(eval_callback)

# Training loop (similar to model.learn in SB3)
for i in range(100):  # Train for 100 iterations
    result = trainer.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

    # Save the model periodically
    if i % 10 == 0:
        checkpoint = trainer.save("a3c_checkpoint")
        print(f"Checkpoint saved at {checkpoint}")

# Save final policy weights
trainer.save("A3C_GRID.pt")

# Load policy weights for testing
trainer.restore("A3C_GRID.pt")

# Testing the model (similar to SB3's evaluation loop)
nb_episode_test = 5
seeds_test_env = (0, 1, 2, 3, 4, 5)    # same size as nb_episode_test
seeds_test_agent = (3, 4, 5, 6, 7)  # same size as nb_episode_test
ts_ep_test =  (0, 1, 2, 3, 4, 5)       # same size as nb_episode_test

ep_infos = {}  # information that will be saved

for ep_test_num in range(nb_episode_test):
    init_obs, init_infos = gym_env.reset(seed=seeds_test_env[ep_test_num])
    trainer.set_weights({"weights_seed": seeds_test_agent[ep_test_num]})
    done = False
    cum_reward = 0
    step_survived = 0
    obs = init_obs
    while not done:
        action = trainer.compute_single_action(obs)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        step_survived += 1
        cum_reward += float(reward)
        done = terminated or truncated
    ep_infos[ep_test_num] = {
        "time serie id": ts_ep_test[ep_test_num],
        "time serie folder": gym_env.envs[0].env.init_env.chronics_handler.get_id(),
        "env seed": seeds_test_env[ep_test_num],
        "agent seed": seeds_test_agent[ep_test_num],
        "steps survived": step_survived,
        "total steps": int(gym_env.envs[0].env.init_env.max_episode_duration()),
        "cum reward": cum_reward
    }

print(json.dumps(ep_infos, indent=4))

# Clean up Ray
ray.shutdown()
