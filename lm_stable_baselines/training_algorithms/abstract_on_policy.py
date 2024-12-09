from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import torch
from stable_baselines3.common.type_aliases import PyTorchObs
from lm_stable_baselines.environments import LanguageModelEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import RolloutReturn
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
from stable_baselines3.common.type_aliases import MaybeCallback 
from copy import deepcopy


class AbstractLMOnPolicy:
    
    def __init__(self, loss_computed_in_forward_pass, batch_size, use_base_model_for_learning=False):
        
        assert all([isinstance(myenv, LanguageModelEnv) for myenv in self.env.envs]), "All environments must \
                                                                                            be of type LanguageModelEnv"
        
        # setting filler tokens
        all_filler_token = [myenv.filler_token for myenv in self.env.envs]
        assert all([filler_token == all_filler_token[0] for filler_token in all_filler_token]), "All environments must \
                                                                                            have the same filler token"
        self.policy.filler_token = all_filler_token[0]
        self.rollout_buffer.set_filler_token(all_filler_token[0])
        self.env.set_filler_token(all_filler_token[0])

        # setting hparams
        self.loss_computed_in_forward_pass = loss_computed_in_forward_pass
        self.policy.predict_values = self.predict_values
        self.batch_size = batch_size
        self.use_base_model_for_learning = use_base_model_for_learning
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> RolloutReturn:
       
        if self.use_base_model_for_learning:
            self.policy.lm.set_adapter(self.name_to_adapter["sampler"])
        
        og_padding_side = self.policy.tokenizer.padding_side
        self.policy.tokenizer.padding_side = "left"
        training = self.policy.training
        self.policy.eval()
        res = super().collect_rollouts(
            env,
            callback,
            rollout_buffer,
            n_rollout_steps,
        )
        self.policy.train(training)
        self.policy.tokenizer.padding_side = og_padding_side
        return res

    # set the forward pass of the base policy
    @staticmethod
    def predict_values(obs: PyTorchObs) -> torch.Tensor:
        # return -1 for all values
        return torch.ones(obs.shape[0]) * 0

    def get_next_observation(self, data):
        next_obs = self.env.envs[0].next_observation_from_observation_and_action(data.observations['input_ids'], data.actions)
        #create the next observation by interacting with the environment and then tokenizing to get input_ids + attention mask
        next_observation = self.policy.tokenizer.pad( 
            {'input_ids': next_obs},
            return_tensors="pt",
            padding=True,
        )
        return next_observation
    
    # def process_sampled_rollouts(self, val_samps): # remove -100 tokens, add 'input_ids' and 'attention_mask' from 'observations' and 'actions' and return the processed samples
    #     keys = ['observations', 'actions']
    #     dict = {}
    #     for key in keys:
    #         # this doens't work, need to get attribute
    #         values = remove_filler_tokens(getattr(val_samps, key), self.policy.filler_token)
    #         values = self.policy.tokenizer.pad(
    #             {'input_ids': values},
    #             return_tensors="pt",
    #             padding=True,
    #         )
    #         dict[key] = values

    def process_sampled_rollouts(self, val_samps): 
        return val_samps
        
    def train(self) -> None:
        self.policy.train()
        if self.use_base_model_for_learning:
            self.policy.lm.set_adapter(self.name_to_adapter["peft_to_train"])
        self.policy.tokenizer.padding_side = "right"
        
        self._update_learning_rate(self.policy.optimizer)
        
        self.rollout_buffer.find_where_advantage_exceeds_threshold(self.rollout_buffer.advantages)
        
        # Your training code here!