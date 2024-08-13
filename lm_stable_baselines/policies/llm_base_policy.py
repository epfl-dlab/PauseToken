from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Type, Optional, Dict, Any, Tuple
from gymnasium import spaces
import torch
import gymnasium as gym
from transformers import PreTrainedModel, GenerationConfig,PreTrainedTokenizer
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor
import warnings
import numpy as np
from lm_stable_baselines.utils import add_filler_tokens

class LLMBasePolicy(BasePolicy):
    """ Base policy for language models. This class is a subclass of BasePolicy and is used to handle language model policies.
    
    :param lr_schedule: Learning rate schedule
    :type lr_schedule: Schedule
    :param lm: Language model
    :type lm: PreTrainedModel
    :param tokenizer: Tokenizer
    :type tokenizer: PreTrainedTokenizer
    :param observation_space: Observation space (if None, it is set to a MultiDiscrete space with the vocab size of the language model)
    :type observation_space: spaces.Space
    :param action_space: Action space (if None, it is set to a Discrete space with the vocab size of the language model)
    :type action_space: spaces.Space
    :param features_extractor_class: Features extractor class
    :type features_extractor_class: Type[BaseFeaturesExtractor]
    :param features_extractor_kwargs: Features extractor keyword arguments
    :type features_extractor_kwargs: Optional[Dict[str, Any]]
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool
    :param optimizer_class: Optimizer class
    :type optimizer_class: torch.optim.Optimizer
    :param optimizer_kwargs: Optimizer keyword arguments
    :type optimizer_kwargs: Optional[Dict[str, Any]]
    :param squash_output: Whether to squash the output
    :type squash_output: bool
    :param filler_token: Filler token
    :type filler_token: int
    :param generation_config: Generation config (Huggingface)
    :type generation_config: GenerationConfig
    :param logit_processor: Logit processor (Huggingface generate function argument)
    :type logit_processor: Any
    :param stopping_criteria: Stopping criteria (Huggingface generate function argument)
    :type stopping_criteria: Any
    :param prefix_allowed_tokens_fn: Prefix allowed tokens function (Huggingface generate function argument)
    :type prefix_allowed_tokens_fn: Any
    :param synced_gpus: Synced GPUs (Huggingface generate function argument)
    :type synced_gpus: Any
    :param assistant_model: Assistant model (Huggingface generate function argument)
    :type assistant_model: Any
    :param streamer: Streamer (Huggingface generate function argument)
    :type streamer: Any
    :param negative_prompt_ids: Negative prompt IDs (Huggingface generate function argument)
    :type negative_prompt_ids: Any
    :param generation_kwargs: Generation keyword arguments (Huggingface generate function argument)
    :type generation_kwargs: Dict[str, Any]
    :param kwargs: Additional keyword arguments
    :type kwargs: Dict[str, Any]
    """
    
    def __init__(
        self,
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
        generation_config=None,
        logit_processor = None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        generation_kwargs = {},
        **kwargs
    ):
        #check if observation_space and action_space are provided and set them to the default values if they are not
        if observation_space is None:
            #check vocab size
            observation_space = spaces.MultiDiscrete([lm.config.vocab_size]* tokenizer.vocab_size, dtype = np.int64) 
        if action_space is None:
            action_space = spaces.MultiDiscrete([lm.config.vocab_size]* tokenizer.vocab_size, dtype = np.int64) 
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        self.lm = lm
        self.tokenizer = tokenizer
        self.filler_token = filler_token

        #args for generation
        self.generation_config = generation_config
        self.logit_processor = logit_processor
        self.stopping_criteria = stopping_criteria
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.synced_gpus = synced_gpus
        self.assistant_model = assistant_model
        self.streamer = streamer
        self.negative_prompt_ids = negative_prompt_ids
        self.generation_kwargs = generation_kwargs
        self.kwargs = kwargs
        self._build(lr_schedule)
        
        
        
    def _build(self, lr_schedule: Schedule):
        """ Build the policy and optimizer
        
        :param lr_schedule: Learning rate schedule
        :type lr_schedule: Schedule
        """
        self.optimizer = self.optimizer_class(
            self.lm.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.lm.eval()
    
    def obs_to_tensor(self, observation: np.ndarray) -> Tuple[PyTorchObs, bool]:    
        """ Convert an observation to a PyTorch tensor
        
        :param observation: Observation
        :type observation: np.ndarray
        :return: Observation tensor and whether the observation is valid
        :rtype: Tuple[PyTorchObs, bool]
        """
        assert isinstance(observation, np.ndarray), "Observation must be a numpy array"
        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, True
        
    def compute_nll_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Compute the negative log likelihood loss
        
        :param logits: Logits
        :type logits: torch.Tensor
        :param labels: Labels
        :type labels: torch.Tensor
        :return: Negative log likelihood loss
        :rtype: torch.Tensor
        """
        shift_lm_logits = logits[..., :-1, :].contiguous()
        shift_lm_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_lm_logits = shift_lm_logits.view(-1, self.lm.config.vocab_size)
        shift_lm_labels = shift_lm_labels.view(-1)
        # Ensure tensors are on the same device
        shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.filler_token)
        return loss_fct(shift_lm_logits, shift_lm_labels)
    
    
    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PyTorchObs:
        if (isinstance(obs, dict) and 'input_ids' in obs and 'attention_mask' not in obs) or isinstance(obs, torch.Tensor):
            warnings.warn("Attention mask not provided, the padding mask will be automatically computed")
            obs_to_pass = obs if isinstance(obs, torch.Tensor) else obs["input_ids"]
            device = obs_to_pass.device
            filler_token_locations = [torch.where(obs==self.filler_token)[0] for obs in obs_to_pass]
            filler_token_locations = [loc[0].item() if len(loc) > 0 else None for loc in filler_token_locations]
            obs_to_pass = [obs[:idx] for idx,obs in zip(filler_token_locations,obs_to_pass)]
            feature = self.tokenizer.pad({"input_ids": obs_to_pass}, return_tensors="pt", padding=True).to(device)
        elif isinstance(obs, dict) and "input_ids" in obs and "attention_mask" in obs:
            feature = obs
        else:
            raise ValueError("Observation type not supported")
        return feature
            
            
    def forward(self, obs: PyTorchObs, action = None) -> torch.Tensor:
        feature = self.extract_features(obs)
        feature = {k: v.to(self.device) for k, v in feature.items()}
        return self.lm(**feature)
    
    def post_predict(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        #remove the input tokens from the output
        actions = outputs[:, inputs.shape[-1]:]
        
        #replace all pad tokens with filler tokens
        actions[actions == self.tokenizer.pad_token_id] = self.filler_token
        
        action_space_dim = self.action_space.shape[0]
        actions = add_filler_tokens(actions, action_space_dim, self.filler_token)
        return actions
    
    def pre_predict(self, observation: PyTorchObs) -> PyTorchObs:
        pass
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False):
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        was_in_training = self.lm.training
        self.lm.eval()
        og_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        feature = self.extract_features(observation)
        inputs = feature["input_ids"]
        self.pre_predict(inputs)
    
        with torch.no_grad():
            outputs = self.lm.generate(
                inputs = feature["input_ids"],
                attention_mask = feature["attention_mask"],
                generation_config=self.generation_config,
                logit_processor = self.logit_processor,
                stopping_criteria= self.stopping_criteria,
                prefix_allowed_tokens_fn= self.prefix_allowed_tokens_fn,
                synced_gpus= self.synced_gpus,
                assistant_model= self.assistant_model,
                streamer= self.streamer,
                negative_prompt_ids= self.negative_prompt_ids,
                **self.generation_kwargs
            )
        
        outputs =  self.post_predict(inputs, outputs)
        if was_in_training:
            self.lm.train()
        self.tokenizer.padding_side = og_padding_side  
        return outputs