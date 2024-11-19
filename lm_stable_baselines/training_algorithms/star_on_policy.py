from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import torch
from stable_baselines3.common.type_aliases import PyTorchObs
from lm_stable_baselines.environments import LanguageModelEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from gymnasium import spaces
from typing import Optional, Union, Dict, Any, List
import numpy as np
from lm_stable_baselines.utils import add_filler_tokens, remove_filler_tokens
from copy import deepcopy
from src.model.components.control_token_wrappers.base_control_token_wrapper import BaseControlTokenWrapper


class STaROnPolicy(OnPolicyAlgorithm):
    
    def __init__(self,*args, loss_computed_in_forward_pass, batch_size ,**kwargs):
        super().__init__(*args, **kwargs)
        assert all([isinstance(myenv, LanguageModelEnv) for myenv in self.env.envs]), "All environments must be of type LanguageModelEnv"
        all_filler_token = [myenv.filler_token for myenv in self.env.envs]
        assert all([filler_token == all_filler_token[0] for filler_token in all_filler_token]), "All environments must have the same filler token"
        self.policy.filler_token = all_filler_token[0]
        self.rollout_buffer.set_filler_token(all_filler_token[0])
        self.env.set_filler_token(all_filler_token[0])
        self.loss_computed_in_forward_pass = loss_computed_in_forward_pass
        self.policy.predict_values = self.predict_values
        self.batch_size = batch_size

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> RolloutReturn:
        
        og_padding_side = self.policy.tokenizer.padding_side
        self.policy.tokenizer.padding_side = "left"
        res = super().collect_rollouts(
            env,
            callback,
            rollout_buffer,
            n_rollout_steps,
        )
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
        
        self._update_learning_rate(self.policy.optimizer)
        nll_losses = []

        self.rollout_buffer.find_where_advantage_exceeds_threshold(self.rollout_buffer.advantages)
        n_batches = self.rollout_buffer.data_size // self.batch_size
        self.policy.tokenizer.padding_side = "right"
        for _ in range(n_batches):

            self._n_updates += 1
            data = self.rollout_buffer.sample_batch(self.batch_size, env=self._vec_normalize_env)
            next_observation = self.get_next_observation(data)
            if self.loss_computed_in_forward_pass:
                labels = next_observation["input_ids"]
                labels_list = list(labels.cpu())
                collated_labels = self.data_collator(labels_list)
                labels = collated_labels["labels"].to(self.device) # check with self.policy.tokenizer.decode(labels[0][labels[0]>0])
            else:
                labels = None
                
            kwargs = {}
            # if isinstance(self.policy.lm, BaseControlTokenWrapper):
            #     kwargs["reduce_mean"] = False

            input_ids = next_observation['input_ids'].to(self.device)

            attention_mask=next_observation['attention_mask'].to(self.device)

            output = self.policy.lm(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            if self.loss_computed_in_forward_pass:
                nll_loss = output.loss
                #if control token model you can also get these losses:
                #control_token_loss = output.ctrl_tok_loss
                #lm_loss = output.lm_loss
            else:
                raise NotImplementedError("To be implemented")
                # nll_loss = self.policy.compute_nll_loss(output.logits, labels)
            

            # # getting log_probs for importance sampling
            # old_log_probs = data.old_log_prob
            
            # mask = labels>0 # get the ids of the tokens that are not padding or the question token
            # labels[labels<0] = 0
            
            # logprobs = torch.log_softmax(output.logits, dim = -1)[:,:-1,:]
            # logprob_actions = torch.gather(logprobs, 2, labels[:,1:].unsqueeze(-1)).squeeze(-1)
            # logprobs = (logprob_actions * mask[:,1:]).sum(dim = 1)
            # breakpoint()
            # ratio = torch.exp(logprobs - old_log_probs).detach() #wrong value at initialization! needs debugging!
            # breakpoint()

            # Compute the loss
            nll_losses.append(nll_loss.item())
            
            self.policy.optimizer.zero_grad()
            
            nll_loss.backward()

            self.policy.optimizer.step()
                    
            
        self.logger.record("train/nll_loss", np.mean(nll_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")