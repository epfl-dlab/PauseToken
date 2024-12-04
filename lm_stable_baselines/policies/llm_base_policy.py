from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Type, Optional, Dict, Any, Tuple, List, Callable
from gymnasium import spaces
import torch
import gymnasium as gym
from transformers import PreTrainedModel, GenerationConfig,PreTrainedTokenizer, LogitsProcessorList, StoppingCriteria
from transformers.generation.streamers import BaseStreamer
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
    :param generation_params: All Generation parameters for the model (used in the generate method of huggingface). This should be a dictionary with both "train" and "test" keys containing the generation parameters for training and testing respectively.
    :type generation_params: Dict[str, Any]
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
        generation_params: Dict[str, Any] = None,
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
        self.lr_schedule = lr_schedule
        self.generation_params = generation_params
        self.kwargs = kwargs
        self._build(lr_schedule)
        self.use_peft_at_inference = False
        
        
        
    def _build(self, lr_schedule: Schedule = None) -> None:
        """ Build the policy and optimizer
        
        :param lr_schedule: Learning rate schedule
        :type lr_schedule: Schedule
        """
        
        if lr_schedule is None:
            lr_schedule = self.lr_schedule
        
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
            obs_to_pass = [ obs[obs != self.filler_token] for obs in obs_to_pass]
            feature = self.tokenizer.pad({"input_ids": obs_to_pass}, return_tensors="pt", padding=True).to(device)
        elif isinstance(obs, dict) and "input_ids" in obs and "attention_mask" in obs:
            feature = obs
        else:
            raise ValueError("Observation type not supported")
        return feature
            
            
    def forward(self, obs: PyTorchObs, labels = None) -> torch.Tensor:
        # feature = self.extract_features(obs)
        # feature = {k: v.to(self.device) for k, v in feature.items()}
        # feature["labels"] = labels.to(self.device) if labels is not None else None

        next_obs, actions, unpadded_actions = self._predict(obs, return_dict= True).values()

        logprobs = torch.log_softmax(self.lm(next_obs).logits, dim = -1)[:, (-unpadded_actions.shape[1]-1):-1, ...]
        mask = (unpadded_actions != self.tokenizer.pad_token_id).float()
        logprob_actions = torch.gather(logprobs, 2, unpadded_actions.unsqueeze(-1)).squeeze(-1)
        logprobs = (logprob_actions * mask).sum(dim = 1)
        values = self.predict_values(next_obs)
        return actions, values, logprobs
    
    
    
    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        raise NotImplementedError
    
    def post_predict(self, inputs: torch.Tensor, outputs: torch.Tensor, return_dict = False) -> torch.Tensor:
        #remove the input tokens from the output
        actions = outputs[:, inputs.shape[-1]:].clone()
        padded_actions = actions.clone()
        #replace all pad tokens with filler tokens
        padded_actions[actions == self.tokenizer.pad_token_id] = self.filler_token
        
        action_space_dim = self.action_space.shape[0]
        padded_actions = add_filler_tokens(padded_actions, action_space_dim, self.filler_token)
        
        if return_dict:
            return {'next_observation':outputs, 'actions': padded_actions, 'unpadded_actions': actions}
        else:
            return padded_actions
    
    def pre_predict(self, feature: PyTorchObs) -> PyTorchObs:
        pass
    
    def disable_peft_at_inference(self):
        self.use_peft_at_inference = False
        
    def enable_peft_at_inference(self):  
        self.use_peft_at_inference = True 
        
    
    def set_generation_cfg(self, name: str):
        assert name in ["train", "test"], "Generation config name must be either 'train' or 'test'"
        self.generation_params_to_use = name
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False, return_dict: bool = False):
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        was_in_training = self.lm.training
        assert self.generation_params_to_use is not None, \
            "You've never set the generation config to use. Please set it using the set_generation_cfg method. Options are 'train' or 'test'"
        generation_params = self.generation_params[self.generation_params_to_use]

        self.lm.eval()
        og_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        feature = self.extract_features(observation)
        inputs = feature["input_ids"]
        self.pre_predict(feature)

        if not self.use_peft_at_inference:
            self.lm.disable_adapter_layers()

        with torch.no_grad():
            outputs = self.lm.generate(
                inputs = inputs,
                attention_mask = feature["attention_mask"],
                tokenizer=self.tokenizer,
                **generation_params,
            )
            
        if not self.use_peft_at_inference:
            self.lm.enable_adapter_layers()
            
        outputs =  self.post_predict(inputs, outputs, return_dict = return_dict)
        if was_in_training:
            self.lm.train()
        self.tokenizer.padding_side = og_padding_side  
        return outputs