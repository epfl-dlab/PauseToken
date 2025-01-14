from lm_stable_baselines.policies.llm_base_policy import LLMBasePolicyValueModel
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

class LLMBasePolicyValueThoughtEmbedModel(LLMBasePolicyValueModel):
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


        super(LLMBasePolicyValueThoughtEmbedModel, self).__init__(
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

        self.thought_embedding_head = hydra.utils.instantiate(kwargs['model']['thought_embedding_head'], _recursive_=False).to(next(self.lm.parameters()).dtype)

        self._build(lr_schedule=lr_schedule)
        
    def evaluate_actions(self, observations, actions):
        """
        Evaluate actions. Used in the training loop to fit the MLP head.
        Returns:
            - values: Predicted state values (for value loss).
            - log_prob: Log probability of the actions (for policy gradient loss).
            - entropy: Entropy of the policy (for exploration bonus).
        """
        # Compute next observations and prepare for LM processing
        next_obs = self.get_next_observation(observations, actions)  # Assuming this is defined elsewhere

        # Compute action log probabilities
        action_start_indices = (observations['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1) - 1
        outputs = self.lm(**next_obs, output_hidden_states=True)
        logits = outputs.logits  # Forward pass through LM
        all_logprobs = torch.log_softmax(logits, dim=-1)  # Convert logits to log-probabilities
        log_probs = self._compute_logprobs(
            all_logprobs[:, :-1, ...], next_obs['input_ids'][:, 1:], action_start_indices
        )

        # Compute entropy
        # action_distribution = torch.distributions.Categorical(logits=logits[:, :-1, ...])
        # entropy = action_distribution.entropy().mean(dim=-1)  # Mean entropy across tokens
        entropy = None # can't approximate it, ppo will simply take log_probs as entropy

        # Compute values, do not use predict values, it will be gradent less, and will do another forward pass through 
        # the LM!
        raw_latent = outputs.hidden_states
        latent = []
        # get observation mask in next_obs
        obs_mask = next_obs['attention_mask'].clone()
        for i in range(next_obs['input_ids'].size(0)):
            obs_mask[i, action_start_indices[i]:] = 0

        for i in range(len(raw_latent)):
            left_padded_embeds, left_padded_mask = self._move_embeddding_padding_to_side(raw_latent[i], obs_mask, left_padding=True)
            latent.append(left_padded_embeds)
        
 
        values = self.value_head(latent, attention_mask=left_padded_mask)

        return values, log_probs, entropy
    

    def predict_values(self, obs) -> torch.Tensor:
        """
        Predict the value of a state.
        Used in the buffer during rollout generation.
        """
        # input questions should be left padded! I am going to take the last hidden state [-1] of the hidden states
        # of the transformer model and pass it through a MLP to get the value of the state!
        assert isinstance(obs, torch.Tensor)
        obs[obs==self.filler_token] = self.tokenizer.pad_token_id

        # if obs is one dimensional, add batch dimension
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        if torch.any(obs[:, -1] == self.tokenizer.pad_token_id):
            # Last token should not be padding token, make sure padding is done from left
            obs = self._move_padding_to_side(obs, left_padding=True)

        attention_mask = (obs != self.tokenizer.pad_token_id).long()
        max_length = attention_mask.sum(dim=1).max().item()
        obs = obs[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        with torch.no_grad():
            output = self.lm(obs, attention_mask=attention_mask,
                             return_dict=True, output_hidden_states=True)

        latent = output['hidden_states']
        values = self.value_head(latent, attention_mask=attention_mask)

        return values.squeeze(-1)  # Squeeze to return 1D tensor for scalar values


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

    def _move_embeddding_padding_to_side(self, obs_embed, padding_mask, left_padding=True):
        """
        Moves padding (denoted by 0) in a batch of sequences to the specified side.

        Args:
            actions (torch.Tensor): Tensor of size (batch_size, sequence_length) 
                                    containing padded sequences with 0 as padding token.
            left_padding (bool): If True, moves padding to the left. If False, moves padding to the right.

        Returns:
            torch.Tensor: Tensor of the same size with padding moved to the specified side.
        """
        batch_size = obs_embed.shape[0]
        length = padding_mask.sum(dim=1).max().item()
        lengths = padding_mask.sum(dim=1)
        
        new_padding_mask = torch.zeros((batch_size, length), dtype=torch.bool, device=obs_embed.device)
        padded_embeds = torch.zeros((batch_size, length, obs_embed.shape[-1]), device=obs_embed.device)
        for i in range(batch_size):
            if left_padding:
                padded_embeds[i, -lengths[i]:] = obs_embed[i, padding_mask[i]==1].clone().detach()
                new_padding_mask[i, -lengths[i]:] = 1
            else:
                padded_embeds[i, :lengths[i]] = obs_embed[i, padding_mask[i]==1].clone().detach()
                new_padding_mask[i, :lengths[i]] = 1     
        return padded_embeds, new_padding_mask