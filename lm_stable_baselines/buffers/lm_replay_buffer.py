from stable_baselines3.common.buffers import ReplayBuffer
import warnings
from typing import Union, List, Dict, Any, Optional
import transformers
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize 
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from lm_stable_baselines.utils import remove_filler_tokens

def double_indexing(array: np.ndarray, idx1: np.ndarray, idx2: Optional[np.ndarray] = None) -> np.ndarray:
    if idx2 is None:
        return array[idx1]
    return array[idx1][idx2]

class LMReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        *args,
        tokenizer: transformers.PreTrainedTokenizer = None,
        advantage_threshold: float = None,
        filler_token = -100, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filler_token = filler_token
        self.observations.fill(self.filler_token)
        self.actions.fill(self.filler_token)
        if not self.optimize_memory_usage:
            self.next_observations.fill(self.filler_token)
        self.tokenizer = tokenizer
        self.advantage_threshold = advantage_threshold
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def set_andvantage_threshold(self, advantage_threshold):
        self.advantage_threshold = advantage_threshold
    
    def set_filler_token(self, filler_token):
        
        self.observations[self.observations == self.filler_token] = filler_token
        self.actions[self.actions == self.filler_token] = filler_token
        if self.optimize_memory_usage:
            self.next_observations[self.next_observations == self.filler_token] = filler_token
        self.filler_token = filler_token
    
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
        
    def sample_batch(self, batch_size, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
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
        
        return self._get_samples(sampled_positions, env)

    def _get_samples(self, idx, env ,padding='right') -> ReplayBufferSamples:
        assert idx[0].shape == idx[1].shape, "The indices must have the same shape"
        
        og_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding
        
        batch_inds, env_indices = idx
        
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        next_obs = self.tokenizer(
            self.tokenizer.batch_decode(
                remove_filler_tokens(next_obs[...,1:], self.filler_token) # remove the first token (the bos token, tokenizer will re-add it)
            ),
            return_tensors="pt", padding=True, truncation=True
        )
        
        obs = self._normalize_obs(self.observations[idx], env)
        
        obs = self.tokenizer(
            self.tokenizer.batch_decode(
                remove_filler_tokens(obs[..., 1:], self.filler_token) # remove the first token (the bos token, tokenizer will re-add it)
            ),
            return_tensors="pt", padding=True, truncation=True
        )
 
        actions = self.tokenizer(
            self.tokenizer.batch_decode(
                remove_filler_tokens(self.actions[idx], self.filler_token) # don't remove the first token (since it's an action, it didn't start with a bos token)
            ),
             return_tensors="pt", padding=True, truncation=True
        )["input_ids"][...,1:] # remove the first token (the bos token, actions should not have it) 

        data = (
            obs,
            actions,
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        
        self.tokenizer.padding_side = og_padding_side
        
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        

    #TODO: I probably can delete this method
    # def add(
    #     self,
    #     obs: np.ndarray,
    #     next_obs: np.ndarray,
    #     action: np.ndarray,
    #     reward: np.ndarray,
    #     done: np.ndarray,
    #     infos: List[Dict[str, Any]],
    # ) -> None:
        
    #     #find sequences where done is True
    #     done_indices = np.where(done)[0]
    #     #check if rewards are above threshold
    #     if self.reward_threshold is not None:
    #         above_thrsh_rewards = np.where(reward[done_indices] > self.reward_threshold)[0]
    #     else:
    #         above_thrsh_rewards = None

    #     if len(done_indices) > 0 and (above_thrsh_rewards is None or len(above_thrsh_rewards) > 0):
    #         finished_rewards = double_indexing(reward, done_indices, above_thrsh_rewards)
    #         finished_obs = double_indexing(obs, done_indices, above_thrsh_rewards)
    #         finished_actions = double_indexing(action, done_indices, above_thrsh_rewards)
            
    #         finished_infos = [infos[i] for i in done_indices]
    #         if above_thrsh_rewards is not None:
    #             finished_infos = [finished_infos[i] for i in above_thrsh_rewards]
            
    #         finished_next_obs = double_indexing(next_obs, done_indices, above_thrsh_rewards)
    #         finished_done = double_indexing(done, done_indices, above_thrsh_rewards)

    #         if len(finished_rewards) == 1:
    #             super().add(finished_obs, finished_next_obs, finished_actions, finished_rewards, finished_done, finished_infos)
    #         elif len(finished_rewards) > 1:
    #             for i in range(len(finished_rewards)):
    #                 super().add(finished_obs[i], finished_next_obs[i], finished_actions[i], finished_rewards[i], finished_done[i], [finished_infos[i]])
    
    # def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    #     # Sample randomly the env idx
    #     env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

    #     if self.optimize_memory_usage:
    #         next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
    #     else:
    #         next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

    #     next_obs = self.tokenizer(
    #         self.tokenizer.batch_decode(
    #             remove_filler_tokens(next_obs[...,1:], self.filler_token) # remove the first token (the bos token, tokenizer will re-add it)
    #         ),
    #         return_tensors="pt", padding=True, truncation=True
    #     )
        
    #     obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        
    #     obs = self.tokenizer(
    #         self.tokenizer.batch_decode(
    #             remove_filler_tokens(obs[..., 1:], self.filler_token) # remove the first token (the bos token, tokenizer will re-add it)
    #         ),
    #         return_tensors="pt", padding=True, truncation=True
    #     )

    #     actions = self.tokenizer(
    #         self.tokenizer.batch_decode(
    #             remove_filler_tokens(self.actions[batch_inds, env_indices, :], self.filler_token) # don't remove the first token (since it's an action, it didn't start with a bos token)
    #         ),
    #          return_tensors="pt", padding=True, truncation=True
    #     )["input_ids"][...,1:] # remove the first token (the bos token, actions should not have it) 

    #     data = (
    #         obs,
    #         actions,
    #         next_obs,
    #         # Only use dones that are not due to timeouts
    #         # deactivated by default (timeouts is initialized as an array of False)
    #         (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
    #         self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
    #     )
    #     return ReplayBufferSamples(*tuple(map(self.to_torch, data)))