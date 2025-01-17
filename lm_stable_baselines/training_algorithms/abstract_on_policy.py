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
from stable_baselines3.common.save_util import save_to_zip_file,load_from_zip_file
import os

from stable_baselines3.common.base_class import SelfBaseAlgorithm
import io
from typing import Type
from stable_baselines3.common.type_aliases import GymEnv
import pathlib
from stable_baselines3.common.utils import get_system_info,check_for_correct_spaces
from stable_baselines3.common.vec_env.patch_gym import _convert_space
import copy

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

    def save(
        self,
        path: str,
        zip_name: Optional[str] = "last_rl_alg_ckpt.zip",
        policy_name: Optional[str] = "last_policy_ckpt.pth",
        exclude: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        pytorch_exclude: Optional[List[str]] = [],
    ):
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = {k: v for k, v in self.__dict__.copy().items() if k != "policy"}
        
        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)
            
        zip_path = os.path.join(path, zip_name)

        save_to_zip_file(zip_path, data=data, params=None, pytorch_variables=None)
        
        policy_path = os.path.join(path, policy_name)
        state_dict = self.policy.optimizer.state_dict()
        torch.save(state_dict, policy_path)

    def load(  # noqa: C901
        self,
        path: str,
        zip_name: str,
        env: Optional[GymEnv] = None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        load_optimizer: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        
        
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        path_to_zip = os.path.join(path, zip_name)
        data, _, _ = load_from_zip_file(
            path_to_zip,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )
        assert data is not None, "No data found in the saved file"
        
        # load parameters
        self.__dict__.update(data)
        self.__dict__.update(kwargs)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if self.use_sde:
            self.policy.reset_noise()  # type: ignore[operator]

    def load_optimizer_state_dict(self, path: str, policy_name: str):
        self.policy._build(lr_schedule=self.lr_schedule)
        path_to_policy = os.path.join(path, policy_name)
        self.policy.optimizer.load_state_dict(torch.load(path_to_policy))

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
        # Your training code here

    def _augment_actions_and_reduce_observations(self, rollout_data):
        next_observation = self.get_next_observation(rollout_data)["input_ids"]
        actions_list = list(next_observation.cpu())
        collated_data = self.data_collator(actions_list)
        # removing the action piece from obersevations
        reduced_observation, augmented_actions = collated_data["input_ids"].to(self.device), collated_data["labels"].to(self.device)
        reduced_observation[augmented_actions != -100] = self.policy.tokenizer.pad_token_id
        reduced_observation = reduced_observation[:, :rollout_data.observations["input_ids"].size(1)]
        reduced_observation = {'input_ids': reduced_observation,
                                    'attention_mask': reduced_observation != self.policy.tokenizer.pad_token_id}
        # attaching the action piece to the actions
        augmented_actions[augmented_actions == -100] = self.policy.tokenizer.pad_token_id
        augmented_actions = self.policy._move_padding_to_side(augmented_actions, left_padding=False)
        max_len = (augmented_actions>0).sum(dim=1).max().item()
        augmented_actions = augmented_actions[:, :max_len]

        observations = reduced_observation
        actions = augmented_actions