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
        self.generation_params = generation_params
        self.kwargs = kwargs
        self._build(lr_schedule)
        self.use_peft_at_inference = False
        
        
        
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
        torch.set_printoptions(sci_mode=False)


    def _move_padding_to_side(self, actions, left_padding=False):
        """
        Moves padding (denoted by 0) in a batch of sequences to the specified side.

        Args:
            actions (torch.Tensor): Tensor of size (batch_size, sequence_length) 
                                    containing padded sequences with 0 as padding token.
            left_padding (bool): If True, moves padding to the left. If False, moves padding to the right.

        Returns:
            torch.Tensor: Tensor of the same size with padding moved to the specified side.
        """
        batch_size, _ = actions.shape
        
        # Create a tensor to store the rearranged output
        rearranged_actions = torch.zeros_like(actions)
        
        for i in range(batch_size):
            # Get non-zero elements (tokens) for each sequence
            tokens = actions[i, actions[i] != 0]
            if left_padding:
                # Place the tokens in the rightmost part of the sequence for left-padding
                rearranged_actions[i, -tokens.size(0):] = tokens
            else:
                # Place the tokens in the leftmost part of the sequence for right-padding
                rearranged_actions[i, :tokens.size(0)] = tokens

        return rearranged_actions

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

        next_obs, actions, padded_actions = self._predict(obs, return_dict= True).values()
        right_padded_next_obs = self._move_padding_to_side(next_obs, left_padding=False)

        # Compute the log probabilities of the actions
        action_start_indecies = (obs != self.filler_token).sum(dim=1) - 1
        logits = self.lm(right_padded_next_obs).logits
        logprobs = torch.log_softmax(logits, dim=-1)
        action_logprobs = self._compute_logprobs(logprobs[:, :-1, ...], right_padded_next_obs[:, 1:], action_start_indecies)

        # # compute the log probabilities of the actions, but on the inputs not shifted!
        # # finding first index which is not padding for each sequence in next_obs
        # non_padding_mask = next_obs != self.tokenizer.pad_token_id
        # indices = torch.arange(next_obs.size(1)).unsqueeze(0).expand(next_obs.size(0), -1).to(self.device)
        # max_number = indices[0, -1] + 1
        # obs_start_id = torch.where(non_padding_mask, indices, torch.full_like(indices, max_number)).min(dim=1).values
        # # obsactions_end_id = torch.where(non_padding_mask, indices, torch.full_like(indices, -1)).max(dim=1).values + 1
        # obs_length = (obs!= self.filler_token).sum(dim=1)
        # logits_on_leftpadded_obs = self.lm(next_obs, attention_mask=(next_obs != self.tokenizer.pad_token_id) ).logits
        # logprobs_on_leftpadded_obs = torch.log_softmax(logits_on_leftpadded_obs, dim=-1)
        # action_logprobs_on_leftpadded_obs = self._compute_logprobs(logprobs_on_leftpadded_obs[:, :-1, ...], next_obs[:, 1:], obs_start_id + obs_length - 1)

        # Compute the values of the actions
        values = self.predict_values(right_padded_next_obs)
        return actions, values, action_logprobs
    
    def _compute_logprobs(self, log_probs, padded_seq, action_start_indecies, action_end_indices=None) -> torch.Tensor:
        # Create index offsets for actions within the concatenated sequence
        if action_end_indices == None:
            action_end_indices = [None] * padded_seq.size(0)
            

        action_mask = torch.zeros_like(padded_seq)
        for i in range(log_probs.size(0)):
            action_mask[i, action_start_indecies[i]:action_end_indices[i]] = 1
        action_mask = action_mask.to(self.device)
        action_mask = action_mask * (padded_seq != self.tokenizer.pad_token_id)

        # Gather log probabilities for the action tokens
        log_probs_seq = torch.gather(log_probs, 2, padded_seq.unsqueeze(-1)).squeeze(-1)
        logprobs = (log_probs_seq * action_mask).sum(dim = 1).to(torch.float16)
        # if logprobs.size(0) <5:
        #     print((padded_seq[0][action_mask[0] == 1]))
        #     print((log_probs_seq * action_mask)[0][action_mask[0]>0])
        # else:
        #     print((padded_seq[1][action_mask[1] == 1]))
        #     print((log_probs_seq * action_mask)[1][action_mask[1]>0])
        return logprobs
        
    
    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        raise NotImplementedError
    
    def post_predict(self, inputs: torch.Tensor, outputs: torch.Tensor, return_dict = False) -> torch.Tensor:
        # for on-policy replay buffer, we need to pad the actions to the max length of the action space, to append to
        # the actions matrix in the buffer.
        #remove the input tokens from the output
        actions = outputs[:, inputs.shape[-1]:].clone()
        filler_token_maxlen_actions = actions.clone()
        #replace all pad tokens with filler tokens
        filler_token_maxlen_actions[actions == self.tokenizer.pad_token_id] = self.filler_token
        
        action_space_dim = self.action_space.shape[0]
        filler_token_maxlen_actions = add_filler_tokens(filler_token_maxlen_actions, action_space_dim, self.filler_token)
        
        if return_dict:
            return {'next_observation':outputs, 'filler_token_maxlen_actions': filler_token_maxlen_actions, 'padded_actions': actions}
        else:
            return filler_token_maxlen_actions
    
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