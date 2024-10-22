from stable_baselines3.common.buffers import RolloutBuffer
import warnings
from typing import Union, List, Dict, Any, Optional
import transformers
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize 
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from lm_stable_baselines.utils import remove_filler_tokens

def double_indexing(array: np.ndarray, idx1: np.ndarray, idx2: Optional[np.ndarray] = None) -> np.ndarray:
    if idx2 is None:
        return array[idx1]
    return array[idx1][idx2]

class LMRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        *args,
        tokenizer: transformers.PreTrainedTokenizer = None,
        advantage_threshold: float = None,
        filler_token = -100, 
        **kwargs
    ):
        self.filler_token = filler_token
        super().__init__(*args, **kwargs)
        self.set_filler_token(filler_token)
        self.tokenizer = tokenizer
        self.advantage_threshold = advantage_threshold
        self.above_threshold_indices = None
        self.data_size = 0
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def reset(self) -> None:
        super().reset()
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.long) + self.filler_token
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.long) +self.filler_token
        self.above_threshold_indices = None
        self.data_size = 0

    
    def set_filler_token(self, filler_token):
        self.filler_token = filler_token
        self.observations.fill(filler_token)
        self.actions.fill(filler_token)


    def to_torch(self, array: Union[np.ndarray, torch.Tensor, transformers.BatchEncoding], copy: bool = True) -> Union[torch.Tensor, transformers.BatchEncoding]:
        if isinstance(array, transformers.BatchEncoding):
            return {k: v.to(self.device) for k,v in array.items()}
        elif isinstance(array, torch.Tensor):
            return array.to(self.device)
        return super().to_torch(array, copy)
    

    def find_where_advantage_exceeds_threshold(self, advantage: np.ndarray) -> None:
        if self.advantage_threshold is None:
            self.advantage_threshold = - np.inf
        self.above_threshold_indices =  np.where(advantage > self.advantage_threshold)
        self.remaining_indices = None
        self.data_size = len(self.above_threshold_indices[0])
    
    def sample_batch(self, batch_size, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        # Initialize remaining indices if it's the first pass or if we've exhausted the dataset
        if self.remaining_indices is None or len(self.remaining_indices[0]) == 0:
            allowed_indices = self.above_threshold_indices if self.above_threshold_indices is not None else np.arange(self.buffer_size)
            # Shuffle the allowed indices
            shuffled_indices = np.random.permutation(np.arange(len(allowed_indices[0])))
            # Store shuffled indices for further sampling
            self.remaining_indices = (allowed_indices[0][shuffled_indices], allowed_indices[1][shuffled_indices])
        
        # Sample from the remaining indices without replacement
        num_remaining = len(self.remaining_indices[0])
        num_to_sample = min(batch_size, num_remaining)

        idx = np.arange(num_remaining)[:num_to_sample]
        sampled_positions = (self.remaining_indices[0][idx], self.remaining_indices[1][idx])

        # Remove the sampled positions from remaining indices
        self.remaining_indices = (
            np.delete(self.remaining_indices[0], idx),
            np.delete(self.remaining_indices[1], idx)
        )
        
        return self.sample_indices(sampled_positions)

    def sample_indices(self, idx, padding='right') -> RolloutBufferSamples:
        assert idx[0].shape == idx[1].shape, "The indices must have the same shape"
        data = (
            self.observations[idx],
            self.actions[idx],
            self.values[idx].flatten(),
            self.log_probs[idx].flatten(),
            self.advantages[idx].flatten(),
            self.returns[idx].flatten(),
        )
        
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
