import json
import time
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from grid2op.gym_compat import ContinuousToDiscreteConverter
#scale the observations/actions

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace
import numpy as np
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import ScalerAttrConverter
from grid2op.gym_compat import ContinuousToDiscreteConverter


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
        # Define attribute names and their corresponding indices in the observation array
        self.obs_attr_to_keep = ["rho", "line_status", "actual_dispatch", "target_dispatch", "time_before_cooldown_line"]
        
        self.obs_indices = {
            "rho": 0,  
            "line_status":1, 
            "actual_dispatch":2, 
            "target_dispatch":3, 
            "time_before_cooldown_line":4    
               
        }
        
        # Set up the observation space
        self._gym_env.observation_space.close()
        ob_space = self._gym_env.observation_space
        # ob_space = BoxGymObsSpace(self._g2op_env.observation_space,
        #                                                 attr_to_keep=self.obs_attr_to_keep)
        
        # ob_space = ob_space.reencode_space("actual_dispatch", 
        #                            ScalerAttrConverter(substract=0.,
        #                                                divide=self._g2op_env.gen_pmax,
        #                                                init_space=ob_space["actual_dispatch"])
        #                           )
        
        # ob_space = ob_space.reencode_space("target_dispatch",
        #                                 ScalerAttrConverter(substract=0.,
        #                                                     divide=self._g2op_env.gen_pmax,
        #                                                     init_space=ob_space["target_dispatch"])
                                            # )

        ob_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                        attr_to_keep=self.obs_attr_to_keep)
        
        self._gym_env.observation_space = ob_space
        # export observation space for the Grid2opEnv
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)
        
        
        
        # Create bins for each observation attribute
        self.obs_bins = {}
        n_bins = 10  
        for attr in self.obs_attr_to_keep:
            obs_low = self._gym_env.observation_space.low[self.obs_indices[attr]]
            obs_high = self._gym_env.observation_space.high[self.obs_indices[attr]]
            # Create bin edges
            self.obs_bins[attr] = np.linspace(obs_low, obs_high, n_bins)

        # new_dim_obs_space = np.sum([np.sum(self._gym_env.observation_space[el].shape).astype(int) 
        #                 for el in self._gym_env.observation_space.spaces])
        # print(f"The new size of the observation space is : "
        #     f"{new_dim_obs_space}")


    def discretise_observation(self, obs, reset = True):

        if reset:
            # Extract the NumPy array from the observation tuple
            values = obs[0]  # This gets the array part of the tuple
            # print(f"Values: {values}")
            discrete_obs = {}
            
            # Initialize a current index to track position in the values array
            current_index = 0
            
            # Iterate through the attributes and discretize their values
            for attr in self.obs_attr_to_keep:
                dim = self._gym_env.observation_space._dims[self.obs_indices[attr]]
                # print(f"Attribute: {attr}, Dimension: {dim}")
                value_slice = values[current_index:current_index + dim]    # Extract the relevant slice for the attribute
                bin_edges = self.obs_bins[attr]  # Get the bin edges for the attribute
                
                # Discretize each value in the slice individually
                digitized_values = np.digitize(value_slice, bin_edges) - 1  # Adjust index to start from 0
                
                # Store the discretized values
                discrete_obs[attr] = digitized_values
                
                # Update the current index for the next attribute
                current_index += dim
            
            # Convert the dictionary of discretized observations back into a regular numpy array
            discrete_array = np.concatenate(list(discrete_obs.values()))
            
            # print(f"Discrete observation as array: {discrete_array}, of size: {discrete_array.size}")
            
            # Return a tuple like the original observation, consisting of the discretized array and obs[1]
            return (discrete_array, obs[1])
        else:
            # Extract the NumPy array from the observation tuple
            values = obs  # This gets the array part of the tuple
            # print(f"Values: {values}")
            discrete_obs = {}
            
            # Initialize a current index to track position in the values array
            current_index = 0
            
            # Iterate through the attributes and discretize their values
            for attr in self.obs_attr_to_keep:
                dim = self._gym_env.observation_space._dims[self.obs_indices[attr]]
                # print(f"Attribute: {attr}, Dimension: {dim}")
                value_slice = values[current_index:current_index + dim]    # Extract the relevant slice for the attribute
                bin_edges = self.obs_bins[attr]  # Get the bin edges for the attribute
                
                # Discretize each value in the slice individually
                digitized_values = np.digitize(value_slice, bin_edges) - 1  # Adjust index to start from 0
                
                # Store the discretized values
                discrete_obs[attr] = digitized_values
                
                # Update the current index for the next attribute
                current_index += dim
            
            # Convert the dictionary of discretized observations back into a regular numpy array
            discrete_array = np.concatenate(list(discrete_obs.values()))
            
            # print(f"Discrete observation as array: {discrete_array}, of size: {discrete_array.size}")
            
            # Return a tuple like the original observation, consisting of the discretized array and obs[1]
            return (discrete_array)









    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started

        self._gym_env.action_space.close()
        act_attr_to_keep =  ["change_line_status", "redispatch"]
        self._gym_env.action_space = self._gym_env.action_space.reencode_space("redispatch",
                                                           ContinuousToDiscreteConverter(nb_bins=11)
                                                           ) 
        self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space,
                                                          attr_to_keep=act_attr_to_keep)
        self.action_space = self._gym_env.action_space


        # print(self.action_space)

        

    def reset(self, seed=None):
        obs = self._gym_env.reset(seed=seed, options=None)  # Ensure this returns a dictionary
       
        discrete_obs = self.discretise_observation(obs, reset = True)
        return discrete_obs
        # return self._gym_env.reset(seed=seed, options=None)


    def step(self, action):
        # Step through the underlying Grid2Op environment
        obs, reward, done, truncated, info = self._gym_env.step(action)
        
        
        # Discretize the observation
        # print(obs)
        discrete_obs = self.discretise_observation(obs, reset = False)
        
        return discrete_obs, reward, done, truncated, info
        # return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()