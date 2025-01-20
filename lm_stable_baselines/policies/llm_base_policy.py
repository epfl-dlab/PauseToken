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
from lm_stable_baselines.utils import remove_filler_tokens

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
        
        self.ft_on_action_only = kwargs.get("ft_on_action_only", False)
        self.per_token_log_prob = kwargs.get("per_token_log_prob", False)
        # dummy value head.
        self.value_head = lambda x, attention_mask: torch.zeros(x[-1].size(0), device=x[-1].device)
        
    def _build(self, lr_schedule: Schedule = None) -> None:
        """ Build the policy and optimizer
        
        :param lr_schedule: Learning rate schedule
        :type lr_schedule: Schedule
        """
        
        if lr_schedule is None:
            lr_schedule = self.lr_schedule

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.lm.eval()
        torch.set_printoptions(sci_mode=False)

    def forward(self, obs: PyTorchObs, labels = None, return_hidden_state=False) -> torch.Tensor:
        """
        Forward pass in the policy. This is used to compute the loss in the training loop.
        and called by the rl algorithm to sample and fill the rollout buffer.
        """
        # generate the actions, get the next_observation=obs+actions and cutaway the excessive pads
        next_obs, actions, _ = self._predict(obs, return_dict= True).values()
        # get as input EXACTLY what the rollout buffer will later give to the policy to be trained.
        values, log_probs, entropy = self.evaluate_actions(obs, actions) 
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, acts):
        """
        Evaluate actions. Used in the training loop to train the policy.
        Returns:
            - values: Predicted state values (for value loss).
            - log_prob: Log probability of the actions (for policy gradient loss).
            - entropy: Entropy of the policy (for exploration bonus).
        """
        # if there are filler tokens in the actions or observations, exchange them with pad tokens (filler tokens 
        # are used to mask the actions in the observations
        observations = self.extract_features(obs)
        actions = self.extract_features(acts)
        # Compute next observations and prepare for LM processing
        next_obs = self.get_next_observation(observations, actions) # Assuming this is defined elsewhere
        # forward pass through the model
        outputs = self.lm(**next_obs, output_hidden_states=True)
        logits = outputs.logits  # Forward pass through LM
        all_logprobs = torch.log_softmax(logits, dim=-1)  # Convert logits to log-probabilities
        
        # if the model is being finetuned on the demonstrations too, then remove those from the obs and
        # append to the actions.
        if not self.ft_on_action_only:
            reduced_observations, augmented_actions = self.augment_actions_reduce_observations(next_obs['input_ids'])
            action_start_indices = (reduced_observations['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1) - 1
        else:
            # Compute action log probabilities
            action_start_indices = (observations['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1) - 1
            
        log_probs = self._compute_logprobs(
            all_logprobs[:, :-1, ...], next_obs['input_ids'][:, 1:], 
            action_start_indices, per_token_log_prob=self.per_token_log_prob
        )

        # Compute values
        raw_latent = outputs.hidden_states
        # get observation mask in next_obs, only attend the obs!
        obs_mask = next_obs['attention_mask'].clone()
        for i in range(next_obs['input_ids'].size(0)):
            obs_mask[i, action_start_indices[i]:] = 0
        values = self.value_forward_pass(raw_latent, obs_mask)

        entropy = - (log_probs * log_probs.exp()).sum(dim=-1).mean()

        return values, log_probs, entropy
    
    def value_forward_pass(self, raw_latent, obs_mask):
        # obs mask is the mask for the observations, whatever they are in the raw_latent...
        # Compute values, 
        # do not use predict values, it will be gradient less, and will do another forward pass through 
        # the LM bc/ it takes observations as inputs.
        latent = []
        for i in range(len(raw_latent)):
            # embeddings should be left padded, so we can query the value...
            left_padded_embeds, left_padded_mask = self._move_obs_embedding_and_attention_mask_to_one_padding_side(raw_latent[i], obs_mask, left_padding=True)
            latent.append(left_padded_embeds)
        # Compute the values of the state! not actions my friend
        values = self.value_head(latent, attention_mask=left_padded_mask)
        return values
    
    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        # return 0 for all values
        # this is only to handle timeouts or last environemnt steps by bootstraping with value function, it's used only in the
        # buffer during rollout generation at the last step when buffer is almost full, or when a sequence is not finished
        # **always without gradients**
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        observations = self.extract_features(obs)
        raw_latents = self.lm(**observations, output_hidden_states=True).hidden_states
        obs_mask = observations['attention_mask']
        values = self.value_forward_pass(raw_latents, obs_mask)
        return values

    def get_next_observation(self, observations, actions):
        #assumption: filler tokens have been removed
        obs_padded_tensor = observations if isinstance(observations, torch.Tensor) else observations["input_ids"]
        actions_padded_tensor = actions if isinstance(actions, torch.Tensor) else actions["input_ids"]

        next_obs = []
        #remove the filler tokens from the actions and observations
        obs_list = remove_filler_tokens(obs_padded_tensor, self.tokenizer.pad_token_id)
        actions_list = remove_filler_tokens(actions_padded_tensor, self.tokenizer.pad_token_id)
        #concatenate the observations and actions
        for obs, action in zip(obs_list, actions_list):
            next_obs.append(torch.cat([obs, action]))
        
        #pad the observations
        new_observations = self.tokenizer.pad({"input_ids": next_obs}, return_tensors="pt", 
                                              padding=True, padding_side="right").to(self.device)

        return new_observations
        
    # def compute_nll_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    #     """ Compute the negative log likelihood loss
        
    #     :param logits: Logits
    #     :type logits: torch.Tensor
    #     :param labels: Labels
    #     :type labels: torch.Tensor
    #     :return: Negative log likelihood loss
    #     :rtype: torch.Tensor
    #     """
    #     shift_lm_logits = logits[..., :-1, :].contiguous()
    #     shift_lm_labels = labels[..., 1:].contiguous()
    #     # Flatten the tokens
    #     shift_lm_logits = shift_lm_logits.view(-1, self.lm.config.vocab_size)
    #     shift_lm_labels = shift_lm_labels.view(-1)
    #     # Ensure tensors are on the same device
    #     shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)
    #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.filler_token)
    #     return loss_fct(shift_lm_logits, shift_lm_labels)
    
    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PyTorchObs:
        if (isinstance(obs, dict) and 'input_ids' in obs and 'attention_mask' not in obs) or isinstance(obs, torch.Tensor):
            # warnings.warn("Attention mask not provided, the padding mask will be automatically computed")
            obs_to_pass = obs if isinstance(obs, torch.Tensor) else obs["input_ids"]
            device = obs_to_pass.device
            obs_to_pass = [ obs[obs != self.filler_token] for obs in obs_to_pass]
            feature = self.tokenizer.pad({"input_ids": obs_to_pass}, return_tensors="pt", padding=True).to(device)

            # empty actions will be float, we need to convert them to long
            if feature["input_ids"].dtype == torch.float32:
                feature["input_ids"] = feature["input_ids"].long()
        elif isinstance(obs, dict) and "input_ids" in obs and "attention_mask" in obs:
            feature = obs
        else:
            raise ValueError("Observation type not supported")
        return feature

    def _compute_logprobs(self, log_probs, padded_seq,
                          action_start_indecies, action_end_indices=None,
                          per_token_log_prob=False) -> torch.Tensor:
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
        logprobs = (log_probs_seq * action_mask).sum(dim = 1).to(log_probs.dtype)
        if per_token_log_prob:
            logprobs = logprobs / action_mask.sum(dim = 1).to(logprobs.dtype)
        
        return logprobs
    
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

        # self.lm.eval()
        og_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        feature = self.extract_features(observation)
        inputs = feature["input_ids"]
        self.pre_predict(feature)

        if not self.use_peft_at_inference:
            self.lm.disable_adapter_layers()

        already_terminated_sequences = (inputs == self.tokenizer.eos_token_id).any(dim = 1)
        
        if not already_terminated_sequences.all(): 
            with torch.no_grad():
                inputs_to_generate = inputs[already_terminated_sequences == False]
                attention_mask = feature["attention_mask"][already_terminated_sequences == False]
                tmp_outputs = self.lm.generate(
                    inputs = inputs_to_generate,
                    attention_mask = attention_mask,
                    tokenizer=self.tokenizer,
                    **generation_params,
                )
            outputs = torch.full((inputs.shape[0], tmp_outputs.shape[1]), self.tokenizer.pad_token_id, dtype = tmp_outputs.dtype, device = tmp_outputs.device)
            outputs[already_terminated_sequences, :inputs.shape[1]] = inputs[already_terminated_sequences]
            outputs[~already_terminated_sequences] = tmp_outputs
            
        else:
            outputs = inputs
            
        if not self.use_peft_at_inference:
            self.lm.enable_adapter_layers()
            
        outputs =  self.post_predict(inputs, outputs, return_dict = return_dict)
        if was_in_training:
            self.lm.train()
        self.tokenizer.padding_side = og_padding_side  
        return outputs
    
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


    def augment_actions_reduce_observations(self, next_observation):
        actions_list = list(next_observation.cpu())
        collated_data = self.data_collator(actions_list)
        # removing the action piece from obersevations
        reduced_observation, augmented_actions = collated_data["input_ids"].to(self.device), collated_data["labels"].to(self.device)
        reduced_observation[augmented_actions != self.filler_token] = self.tokenizer.pad_token_id
        max_obs_len = (reduced_observation>0).sum(dim=1).max().item()
        reduced_observation = reduced_observation[:, :max_obs_len]
        reduced_observation = {'input_ids': reduced_observation,
                                    'attention_mask': reduced_observation != self.tokenizer.pad_token_id}
        # attaching the action piece to the actions
        augmented_actions[augmented_actions == self.filler_token] = self.tokenizer.pad_token_id
        augmented_actions = self._move_padding_to_side(augmented_actions, left_padding=False)
        max_len = (augmented_actions>0).sum(dim=1).max().item()
        augmented_actions = augmented_actions[:, :max_len]

        return reduced_observation, augmented_actions
    
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
    
    def _move_obs_embedding_and_attention_mask_to_one_padding_side(self, obs_embed, padding_mask, left_padding=True):
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
    

    def save_additional_modules(self, save_path):
        """
        Save additional modules (value head) to the save path.
        """
        pass

    def load_additional_modules(self, load_path):
        """
        Load additional modules (value head) from the load path.
        """
        pass