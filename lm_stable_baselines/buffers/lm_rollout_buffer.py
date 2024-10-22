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
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.long)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.long)
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
        self.above_threshold_indices =  np.where(advantage > self.advantage_threshold)
        self.data_size = len(self.above_threshold_indices[0])


    def sample_batch(self, batch_size, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        # Get the positions of the allowed indices (where the matrix is 1)
        allowed_indices = self.above_threshold_indices if self.above_threshold_indices is not None else np.arange(self.buffer_size)

        # Sample randomly from the allowed indices
        idx = np.random.choice(len(allowed_indices), size=batch_size, replace=True)
        sampled_positions = (allowed_indices[0][idx], allowed_indices[1][idx])
        
        obs = self.observations[sampled_positions]
        
        obs = self.tokenizer(
            self.tokenizer.batch_decode(
                remove_filler_tokens(obs[..., 1:].long(), self.filler_token)  # remove the first token (the bos token, tokenizer will re-add it)
            ),
            return_tensors="pt", padding=True, truncation=True
        )

        actions = self.tokenizer(
            self.tokenizer.batch_decode(
                remove_filler_tokens(self.actions[sampled_positions], self.filler_token)  # don't remove the first token (since it's an action, it didn't start with a bos token)
            ),
            return_tensors="pt", padding=True, truncation=True
        )["input_ids"][..., 1:]  # remove the first token (the bos token, actions should not have it)

        data = (
            self.observations[sampled_positions],
            self.actions[sampled_positions],
            self.values[sampled_positions].flatten(),
            self.log_probs[sampled_positions].flatten(),
            self.advantages[sampled_positions].flatten(),
            self.returns[sampled_positions].flatten(),
        )
        
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

