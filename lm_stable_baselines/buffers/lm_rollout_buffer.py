from stable_baselines3.common.buffers import RolloutBuffer
import warnings
from typing import Union, List, Dict, Any, Optional
import transformers
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize 
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from lm_stable_baselines.utils import remove_filler_tokens
# dataloader wrapper for the rollout buffer
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Tuple

def double_indexing(array: np.ndarray, idx1: np.ndarray, idx2: Optional[np.ndarray] = None) -> np.ndarray:
    if idx2 is None:
        return array[idx1]
    return array[idx1][idx2]

class LMRolloutBuffer(RolloutBuffer, IterableDataset):
    def __init__(
        self,
        *args,
        tokenizer: transformers.PreTrainedTokenizer = None,
        advantage_threshold: float = None,
        filler_token = -100, 
        **kwargs
    ):
        self.filler_token = filler_token
        # taking care of rollout buffer arguments
        rollout_buffer_kwargs = {k: kwargs[k] for k in kwargs if k in RolloutBuffer.__init__.__code__.co_varnames}
        # RolloutBuffer.__init__(self, *args, **rollout_buffer_kwargs)
        super().__init__(*args, **rollout_buffer_kwargs)
        self.set_filler_token(filler_token)
        self.tokenizer = tokenizer
        self.advantage_threshold = advantage_threshold
        self.above_threshold_indices = None
        self.data_size = 0
        self.model_dtype = kwargs.get("model_dtype", torch.float16)
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        # convert log_prob, value to float 32 (problematic when model is in float 16)
        log_prob = log_prob.float()
        value = value.float()
        super().add(obs, action, reward, episode_start, value, log_prob)
    
    def reset(self) -> None:
        super().reset()
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.int64) + self.filler_token
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int64) + self.filler_token
        self.above_threshold_indices = None
        self.data_size = 0
    
    def set_filler_token(self, filler_token):
        self.filler_token = filler_token
        self.observations.fill(filler_token)
        self.actions.fill(filler_token)
    
    def find_where_advantage_exceeds_threshold(self, advantage: np.ndarray, override_advantage_threshold = None) -> None:
        if override_advantage_threshold is not None:
            advantage_threshold = override_advantage_threshold
        
        elif self.advantage_threshold is None:
            self.advantage_threshold = - np.inf
            advantage_threshold = self.advantage_threshold
            
        else:
            advantage_threshold = self.advantage_threshold
        
        self.above_threshold_indices =  np.where(advantage > advantage_threshold)
        if not self.full:
            filled_positions = np.where(self.above_threshold_indices[0] < self.pos)
            self.above_threshold_indices = (self.above_threshold_indices[0][filled_positions], self.above_threshold_indices[1][filled_positions])       
        self.remaining_indices = None
        self.data_size = len(self.above_threshold_indices[0])
    
    def sample_batch(self, batch_size, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        # Initialize remaining indices if it's the first pass or if we've exhausted the dataset
        allowed_indices = self.above_threshold_indices if self.above_threshold_indices is not None else np.arange(self.buffer_size)
        # Shuffle the allowed indices
        shuffled_indices = np.random.permutation(np.arange(len(allowed_indices[0])))
        
        for i in range(0, len(shuffled_indices), batch_size):
            num_to_sample = min(batch_size, len(shuffled_indices) - i)
            indices = shuffled_indices[i:i + num_to_sample]
            idx = (allowed_indices[0][indices][0], allowed_indices[1][indices][0])
            yield self._get_samples(idx, env)

    def __iter__(self) -> Iterator[Tuple]:
        return self.sample_batch(1)
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        if last_values.dtype == torch.bfloat16:
            last_values = last_values.float()
        super().compute_returns_and_advantage(last_values, dones)
    
    def _get_samples(self, batch_inds, env: Optional[VecNormalize] = None, padding='right') -> RolloutBufferSamples:
        # obs = self.remove_filler_tokens_and_pad(self.observations, batch_inds)
        # actions = self.remove_filler_tokens_and_pad(self.actions, batch_inds)["input_ids"]
        # if model dtype is bfloat16, convert values to bfloat16
        if self.model_dtype == 'torch.bfloat16':
            # map to bfloat16
            values = self.values[batch_inds].bfloat16()
            log_probs = self.log_probs[batch_inds]
            advantages = self.advantages[batch_inds]
            returns = self.returns[batch_inds]
        else:
            obs = self.observations[batch_inds]
            obs[obs==self.filler_token] = self.tokenizer.pad_token_id
            actions = self.actions[batch_inds]
            actions[actions==self.filler_token] = self.tokenizer.pad_token_id
            values = self.values[batch_inds]
            log_probs = self.log_probs[batch_inds]
            advantages = self.advantages[batch_inds]
            returns = self.returns[batch_inds]
        data = (
            obs,
            actions,
            values.flatten(),
            log_probs.flatten(),
            advantages.flatten(),
            returns.flatten(),
        )

        return RolloutBufferSamples(*tuple(data))
        
    def remove_filler_tokens_and_pad(self, tensor, batch_inds,):
        tensor_list = remove_filler_tokens(tensor[batch_inds], self.filler_token)
        max_len = max([len(t) for t in tensor_list])
        tensor_tensor = torch.ones(len(tensor_list), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
        for i, t in enumerate(tensor_list):
            tensor_tensor[i, :len(t)] = torch.tensor(t)
        #NICKY: Changed this to clone().detach() due to pytorch warnings
        output_tensor = {"input_ids": tensor_tensor, 
                         "attention_mask": (tensor_tensor != self.tokenizer.pad_token_id).long()}
        return output_tensor
    


def dataloader_from_buffer(buffer, batch_size):
    """Initialize the Replay Buffer dataset used for retrieving experiences."""
    dataset = buffer
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )
    return dataloader