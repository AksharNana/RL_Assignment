import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch as th
from typing import Any, Dict, List, Tuple, Union, Optional
from stable_baselines3.common.vec_env import VecNormalize 

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        
        self.alpha = 0.4
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        max_prio = self.priorities.max() if self.full else 1.0
        super().add(obs, next_obs, action, reward, done, infos)  # Call the parent add method
        
        # Update the priority of the new sample
        self.priorities[self.pos] = max_prio

    def sample(self, batch_size: int, beta: float = 0.4, env: Optional[VecNormalize] = None):
        if self.size() == 0:  # Use self.size() method to get buffer size
            raise ValueError("Buffer is empty. Cannot sample.")

        # Use the correct size for probabilities
        prios = self.priorities[:self.size()]  # Get only the filled priorities

        # Calculate probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(self.size(), batch_size, p=probs)

        # Retrieve the actual data from the buffer (parent class handles this)
        replay_data = super()._get_samples(indices, env=env)
      

        total = self.size()
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Return replay data along with weights and indices
        return replay_data  #, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray) -> None:
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
