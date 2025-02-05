from lm_stable_baselines.policies.llm_base_policy import LLMBasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
import torch
import torch.nn as nn
import hydra

from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Type, Optional, Dict, Any
from gymnasium import spaces

from transformers import PreTrainedModel,PreTrainedTokenizer
from stable_baselines3.common.type_aliases import Schedule
import os

class LLMBasePolicyValueModel(LLMBasePolicy):
    """
    Base class for all LLMs with a value head.
    """
    
    def __init__(self,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        lr_schedule: Schedule = None,
        tokenizer: PreTrainedTokenizer = None,
        lm: PreTrainedModel = None,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        squash_output: bool = False,
        filler_token: int = -100,
        generation_params: Dict[str, Any] = None,
        **kwargs):


        super(LLMBasePolicyValueModel, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            tokenizer=tokenizer,
            lm=lm,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            filler_token=filler_token,
            generation_params=generation_params,
            **kwargs
        )

        self.value_head = hydra.utils.instantiate(kwargs['model']['value_head'], _recursive_=False).to(next(self.lm.parameters()).dtype)
        print(next(self.lm.parameters()).dtype)

        self._build(lr_schedule=lr_schedule)

    def save_additional_modules(self, save_path):
        """
        Save additional modules (value head) to the save path.
        """
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "value_head.pth")
        torch.save(self.value_head.state_dict(), filename)

    def load_additional_modules(self, load_path):
        """
        Load additional modules (value head) from the load path.
        """
        filename = os.path.join(load_path, "value_head.pth")
        self.value_head.load_state_dict(torch.load(filename))

    # def predict_values(self, obs) -> torch.Tensor:
    #     """
    #     Predict the value of a state.
    #     Used in the buffer during rollout generation.
    #     """
    #     # input questions should be left padded! I am going to take the last hidden state [-1] of the hidden states
    #     # of the transformer model and pass it through a MLP to get the value of the state!
    #     assert isinstance(obs, torch.Tensor)
    #     obs[obs==self.filler_token] = self.tokenizer.pad_token_id

    #     # if obs is one dimensional, add batch dimension
    #     if len(obs.shape) == 1:
    #         obs = obs.unsqueeze(0)

    #     if torch.any(obs[:, -1] == self.tokenizer.pad_token_id):
    #         # Last token should not be padding token, make sure padding is done from left
    #         obs = self._move_padding_to_side(obs, left_padding=True)

    #     attention_mask = (obs != self.tokenizer.pad_token_id).long()
    #     max_length = attention_mask.sum(dim=1).max().item()
    #     obs = obs[:, -max_length:]
    #     attention_mask = attention_mask[:, -max_length:]
    #     with torch.no_grad():
    #         output = self.lm(obs, attention_mask=attention_mask,
    #                          return_dict=True, output_hidden_states=True)

    #     latent = output['hidden_states']
    #     values = self.value_head(latent, attention_mask=attention_mask)
    #     return values.squeeze(-1)  # Squeeze to return 1D tensor for scalar values



