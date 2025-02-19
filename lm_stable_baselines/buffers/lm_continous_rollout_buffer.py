from lm_stable_baselines.buffers.lm_rollout_buffer import LMRolloutBuffer
import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize 
import torch
from lm_stable_baselines.utils import unhash_ids_and_hidden_states, hash_ids_and_hidden_states, pad_hidden_states
from stable_baselines3.common.type_aliases import RolloutBufferSamples




class LMContinousRolloutBuffer(LMRolloutBuffer):

    def remove_filler_tokens_and_pad(self, tensor, batch_inds,):
        
        features = unhash_ids_and_hidden_states(tensor[batch_inds])
        
        obs_to_pass = features["input_ids"]
        obs_to_pass = [ obs[obs != self.filler_token] for obs in obs_to_pass]
        feature = self.tokenizer.pad({"input_ids": obs_to_pass}, return_tensors="pt", padding=True, padding_side="right")
        if feature["input_ids"].dtype == torch.float32:
            feature["input_ids"] = feature["input_ids"].long()

        padding_side = "right"
        if (features["hidden_states"] == self.filler_token).all():
            feature["thought_hidden_states"] = None
        else:
            feature["thought_hidden_states"] = pad_hidden_states(
                hidden_states=features["hidden_states"],
                attention_mask=feature["attention_mask"],
                filler_token=0,
                padding_side=padding_side
            )
        return feature
    
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
        actions = self.remove_filler_tokens_and_pad(self.actions.reshape(-1, self.action_space.shape[0], self.action_space.shape[1]), batch_inds)
        # if model dtype is bfloat16, convert values to bfloat16
        if self.model_dtype == 'torch.bfloat16':
            # make them bfloat16 tensort
            # values = torch.tensor(self.values[batch_inds], dtype=torch.bfloat16)
            # log_probs = torch.tensor(self.log_probs[batch_inds], dtype=torch.bfloat16)
            # advantages = torch.tensor(self.advantages[batch_inds], dtype=torch.bfloat16)
            values = self.values[batch_inds]
            log_probs = self.log_probs[batch_inds]
            advantages = self.advantages[batch_inds]
            
            returns = torch.tensor(self.returns[batch_inds], dtype=torch.bfloat16)
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
    
 