from lm_stable_baselines.policies.llm_base_policy import LLMBasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
import torch
import torch.nn as nn

from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Type, Optional, Dict, Any
from gymnasium import spaces

from transformers import PreTrainedModel,PreTrainedTokenizer
from stable_baselines3.common.type_aliases import Schedule


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
        hidden_size: int = 768,
        value_head_hidden_dim: int = 64,
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

        # Define the value head (MLP on top of the transformer)
        self.MLP_value_head = nn.Sequential(
            nn.Linear(hidden_size, value_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_head_hidden_dim, 1),
        )

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
        right_padded_next_obs = self._move_padding_to_side(next_obs, left_padding=False)

        # Compute action log probabilities
        action_start_indices = (observations != self.filler_token).sum(dim=1) - 1
        logits = self.lm(right_padded_next_obs).logits  # Forward pass through LM
        all_logprobs = torch.log_softmax(logits, dim=-1)  # Convert logits to log-probabilities
        log_probs = self._compute_logprobs(
            all_logprobs[:, :-1, ...], right_padded_next_obs[:, 1:], action_start_indices
        )

        # Compute entropy
        action_distribution = torch.distributions.Categorical(logits=logits[:, :-1, ...])
        entropy = action_distribution.entropy().mean(dim=-1)  # Mean entropy across tokens

        # Compute values
        values = self.predict_values(right_padded_next_obs)

        return values, log_probs, entropy


    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Predict the value of a state.
        Used in the buffer during rollout generation.
        """
        with torch.no_grad():
            latent = self.lm(obs).last_hidden_state[:, 0, :]  # Use CLS token (or equivalent) for value prediction

        values = self.MLP_value_head(latent)
        return values.squeeze(-1)  # Squeeze to return 1D tensor for scalar values



