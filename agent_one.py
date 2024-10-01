import json
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace

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
        obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p"]
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                         attr_to_keep=obs_attr_to_keep
                                                         )

        # export observation space for the Grid2opEnv
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started

        self._gym_env.action_space.close()
        act_attr_to_keep =  ["set_line_status_simple", "set_bus"]
        self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space,
                                                      attr_to_keep=act_attr_to_keep)
        self.action_space = Discrete(self._gym_env.action_space.n)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


from stable_baselines3.common.callbacks import BaseCallback

# Custom callback class to print progress
class ProgressCallback(BaseCallback):
    def __init__(self, verbose=1, print_freq=100):
        super(ProgressCallback, self).__init__(verbose)
        self.print_freq = print_freq  # Frequency at which to print the progress

    def _on_step(self) -> bool:
        # This will be called after every step in the environment
        if self.n_calls % self.print_freq == 0:
            # n_calls is the number of times the callback has been called
            print(f"Step: {self.n_calls}, Timesteps: {self.num_timesteps}, Reward: {self.locals['rewards']}")
        return True  # To continue training

from stable_baselines3 import PPO

env = Gym2OpEnv()
gym_env = env._gym_env
agent = PPO("MlpPolicy", gym_env, verbose=0)

progress_callback = ProgressCallback(print_freq=1)
agent.learn(total_timesteps=10, callback=progress_callback)

nb_episode_test = 2
seeds_test_env = (0, 1)    # same size as nb_episode_test
seeds_test_agent = (3, 4)  # same size as nb_episode_test
ts_ep_test =  (0, 1)       # same size as nb_episode_test

ep_infos = {}  # information that will be saved


for ep_test_num in range(nb_episode_test):
    init_obs, init_infos = gym_env.reset(seed=seeds_test_env[ep_test_num],
                                         options={"time serie id": ts_ep_test[ep_test_num]})
    agent.set_random_seed(seeds_test_agent[ep_test_num])
    done = False
    cum_reward = 0
    step_survived = 0
    obs = init_obs
    while not done:
        act, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(act)
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
