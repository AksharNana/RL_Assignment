import json
import time
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace, ScalerAttrConverter, ContinuousToDiscreteConverter

from lightsim2grid import LightSimBackend


# Gymnasium environment wrapper around Grid2Op environment
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

        # Base observation space attributes that will be kept

        obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "gen_p", "gen_q", "load_p", "load_q",
                    "actual_dispatch", "rho", "line_status", "storage_power", "storage_charge"]

        observation_space = self._gym_env.observation_space
        self._gym_env.observation_space.close()
        gen_pmax = self._g2op_env.gen_pmax

        # Rescaling attributes in the observation space to be roughly from 0 to 1

        
        # observation_space = observation_space.reencode_space("actual_dispatch", 
        #                                 ScalerAttrConverter(substract=0.,
        #                                                     divide=gen_pmax,
        #                                                     init_space=observation_space["actual_dispatch"])
        #                                 )

        # observation_space = observation_space.reencode_space("load_p",
        #                                 ScalerAttrConverter(substract=obs_gym["load_p"],
        #                                                     divide=0.5 * obs_gym["load_p"]
        #                                                     )
        #                                 )
        obs_gym, info = self._gym_env.reset()
        self._gym_env.observation_space = observation_space
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                   attr_to_keep=obs_attr_to_keep,
                                                   divide={"gen_p": self._g2op_env.gen_pmax,
                                                  "load_p": obs_gym["load_p"],
                                                  "actual_dispatch": self._g2op_env.gen_pmax},)
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

        # Base action space attributes that will be kept

        act_attr_to_keep = ["redispatch"]
        self._gym_env.action_space = action_space
        self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space,
                                                           attr_to_keep=act_attr_to_keep)
        self.action_space = Box(shape=self._gym_env.action_space.shape,
                                     low=self._gym_env.action_space.low,
                                     high=self._gym_env.action_space.high)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)
  
    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

env = Gym2OpEnv()

gym_env = DummyVecEnv(
    [
        lambda: Monitor(
           env._gym_env
        )
    ]
)

vec_env = make_vec_env(lambda : Gym2OpEnv(), n_envs=4)

# Create a PPO model, and pass in the vectorized gym environment


model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate= 0.0001,  # Adjusted learning rate for stability
    batch_size=256,        # Increased batch size for better updates
    gamma=0.95,            # Adjusted discount factor for longer-term rewards
    ent_coef=0.05,       # Keep auto, but monitor its decay
    verbose=1,
    clip_range=0.1,
    tensorboard_log="./tb_logs/",
    device="cuda",
)

# Specify callbacks for evaluation 

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./models",
    log_path="./logs",
    deterministic=True,
    eval_freq=10000,
)

callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks


# # Load policy weights
# model.load("model.zip")


# Train for a certain number of timesteps
model.learn(
    total_timesteps=500000, tb_log_name="PPO_TRAIN" + str(time.time()), **kwargs
)

# Save policy weights
model.save("model.zip")


# Given evaluation code

max_steps = 10000
count_failedActions = 0

print("#####################")
print("# OBSERVATION SPACE #")
print("#####################")
print(env.observation_space)
print("#####################\n")

print("#####################")
print("#   ACTION SPACE    #")
print("#####################")
print(env.action_space)
print("#####################\n\n")

curr_step = 0
curr_return = 0

is_done = False
obs, info = env.reset()
print(f"step = {curr_step} (reset):")
print(f"\t obs = {obs}")
print(f"\t info = {info}\n\n")

while not is_done and curr_step < max_steps:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    curr_step += 1
    curr_return += reward
    is_done = terminated or truncated

    print(f"step = {curr_step}: ")
    # print(f"\t obs = {obs}")
    print(f"\t reward = {reward}")
    print(f"\t terminated = {terminated}")
    print(f"\t truncated = {truncated}")
    print(f"\t info = {info}")

    # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
    # Invalid actions are replaced with 'do nothing' action
    is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
    print(f"\t is action valid = {is_action_valid}")
    if not is_action_valid:
        print(f"\t\t reason = {info['exception']}")
        count_failedActions  += 1
    print("\n")

print("###########")
print("# SUMMARY #")
print("###########")
print(f"return = {curr_return}")
print(f"total steps = {curr_step}")
print(f"Number of failed actions = {count_failedActions}")
print("###########")


# Custom evaluation code, evaluate over 10 episodes until termination. Stores the results in a ep_infos object, print as json.

nb_episode_test = 10
ep_infos = {}  # information that will be saved

total_cum_reward = 0
total_steps_survived = 0

for ep_test_num in range(nb_episode_test):
    init_obs, init_infos = env.reset()
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

    ep_infos[ep_test_num] = {"time serie id": ep_test_num,
                             "time serie folder": env._gym_env.init_env.chronics_handler.get_id(),
                             "steps survived": step_survived,
                             "total steps": int(env._gym_env.init_env.max_episode_duration()),
                             "cum reward": cum_reward}
print(json.dumps(ep_infos, indent=10))

avg_rew = total_cum_reward / 10
avg_step = total_steps_survived / 10
print("###########")
print("# SUMMARY #")
print("###########")
print(f"Average reward = {avg_rew}")
print(f"total steps survived= {avg_step}")
print("###########")