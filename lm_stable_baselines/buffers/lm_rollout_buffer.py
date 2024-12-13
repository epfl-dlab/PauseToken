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
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int64) +self.filler_token
        self.above_threshold_indices = None
        self.data_size = 0

    
    def set_filler_token(self, filler_token):
        self.filler_token = filler_token
        self.observations.fill(filler_token)
        self.actions.fill(filler_token)


    def to_torch(self, array: Union[np.ndarray, torch.Tensor, transformers.BatchEncoding, dict], copy: bool = True) -> Union[torch.Tensor, transformers.BatchEncoding]:
        if isinstance(array, transformers.BatchEncoding,) or isinstance(array, dict):
            return {k: v.to(self.device) for k,v in array.items()}
        elif isinstance(array, torch.Tensor):
            return array.to(self.device)
        return super().to_torch(array, copy)
    

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
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        if last_values.dtype == torch.bfloat16:
            last_values = last_values.float()
        super().compute_returns_and_advantage(last_values, dones)

    
    def _get_samples(self, batch_inds, env: Optional[VecNormalize] = None, padding='right') -> RolloutBufferSamples:
        
        # obs = self.tokenizer(
        #     self.tokenizer.batch_decode(
        #         remove_filler_tokens(self.observations[batch_inds][..., 1:], self.filler_token) # remove the first token (the bos token, tokenizer will re-add it)
        #     ),
        #     return_tensors="pt", padding=True, truncation=True
        # )
 
        # actions = self.tokenizer(
        #     self.tokenizer.batch_decode(
        #         remove_filler_tokens(self.actions[batch_inds], self.filler_token) # don't remove the first token (since it's an action, it didn't start with a bos token)
        #     ),
        #      return_tensors="pt", padding=True, truncation=True
        # )["input_ids"][...,1:] # remove the first token (the bos token, actions should not have it) 

        # this messes up by retokenizing, and same text can be tokenized differently
        # obs_list = remove_filler_tokens(self.observations[batch_inds][..., 0:], self.filler_token) # No tokenizer, keep the BOS
        # max_obs_len = max([len(obss) for obss in obs_list])
        # obs_tensor = torch.ones(len(obs_list), max_obs_len, dtype=torch.long) * self.tokenizer.pad_token_id
        # for i, obs in enumerate(obs_list):
        #     obs_tensor[i, :len(obs)] = torch.tensor(obs)
        # obs = {"input_ids": obs_tensor, "attention_mask": torch.tensor(obs_tensor != self.tokenizer.pad_token_id).long()}
        obs = self.remove_filler_tokens_and_pad(self.observations, batch_inds)

        # actions_list = remove_filler_tokens(self.actions[batch_inds], self.filler_token)
        # max_actions_len = max([len(actions) for actions in actions_list])
        # actions_tensor = torch.ones(len(actions_list), max_actions_len, dtype=torch.long) * self.tokenizer.pad_token_id
        # for i, actions in enumerate(actions_list):
        #     actions_tensor[i, :len(actions)] = torch.tensor(actions)
        # actions = actions_tensor
        actions = self.remove_filler_tokens_and_pad(self.actions, batch_inds)["input_ids"]
        # if model dtype is bfloat16, convert values to bfloat16
        if self.model_dtype == torch.bfloat16:
            values = self.values[batch_inds].to(torch.bfloat16)
            log_probs = self.log_probs[batch_inds].to(torch.bfloat16)
            advantages = self.advantages[batch_inds].to(torch.bfloat16)
            returns = self.returns[batch_inds].to(torch.bfloat16)
        else:
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

        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
        
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