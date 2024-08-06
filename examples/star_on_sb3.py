# import re
# import numpy as np
# import torch
# from typing import Union, List
# from transformers import PreTrainedTokenizer
# ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
# INVALID_ANS = "[invalid]"

# def remove_filler_tokens(obs: torch.Tensor, filler_token: int) -> Union[torch.Tensor, List[torch.Tensor]]:
#     """ Remove filler tokens from the obs tensor. Function usually used before padding
    
#     :param obs: Observation tensor
#     :type obs: torch.Tensor
#     :param filler_token: Filler token
#     :type filler_token: int
#     :return: Observation tensor without filler tokens, returns either a 2D tensor or a list of 1D tensors
#     :rtype: Union[torch.Tensor, List[torch.Tensor]]
#     """
#     #check for any filler tokens
#     if not (obs == filler_token).any():
#         return obs
    
#     shape = obs.shape
#     #if it is a 1D tensor we can filter it directly
#     if len(shape) == 1:
#         return obs[obs != filler_token].reshape(-1,1)
#     #If it is a 2D tensor we have to filter each row and return a list of 1D tensors
#     return [ob[ob != filler_token] for ob in obs]


# def add_filler_tokens(array: np.array, max_tokens: int, filler_token: int)-> np.array:
#     """ Add filler tokens to the array to make it of length max_tokens
    
#     :param array: Array to add filler tokens to
#     :type array: np.array
#     :param max_tokens: Maximum number of tokens
#     :type max_tokens: int
#     :param filler_token: Filler token
#     :type filler_token: int
#     :return: Array with filler tokens
#     :rtype: np.array
#     """
#     if array.shape[-1] < max_tokens:
#         return np.concatenate([array, np.array([filler_token] * (max_tokens - array.shape[-1]))])
#     return array

# def extract_answer(completion: str) -> str:
#     """ Extracts the answer from the completion following the GSM8K dataset format
    
#     :param completion: Completion
#     :type completion: str
#     :return: Extracted answer
#     :rtype  str
#     """
#     match = ANS_RE.search(completion)
#     if match:
#         match_str = match.group(1).strip()
#         match_str = match_str.replace(",", "")
#         return match_str
#     else:
#         return INVALID_ANS

# def is_correct(model_completion: str, gt_example: str) -> bool:
#     """ Check if the model completion is correct given the ground truth example. Completions must be in the GSM8K dataset format
    
#     :param model_completion: Model completion
#     :type model_completion: str
    
#     """
#     gt_answer = extract_answer(gt_example["answer"])
#     assert gt_answer != INVALID_ANS, \
#         f"Ground truth answer is invalid and doesn't follow the GSM8K formate, your ground truth answer is {gt_example['answer']}"
#     return extract_answer(model_completion) == gt_answer

# def strip_special_tokens(input_ids: Union[int, List[int], np.ndarray, torch.Tensor], tokenizer: PreTrainedTokenizer) -> str:
#     """ Strip special tokens from a text
    
#     :param input_ids: Input ids
#     :type text: Union[int, List[int], np.ndarray, torch.Tensor]
#     :param tokenizer: Tokenizer
#     :type tokenizer: transformers.PreTrainedTokenizer
#     :return: Text without special tokens
#     :rtype: str
#     """
#     return tokenizer.decode(input_ids, skip_special_tokens=True)



# import sys
# from typing import List,Dict,Union
# import torch


# class AbstractReward:
#     """ Abstract class for reward functions. This class should be subclassed and the reward_fn method should be overriden. 
#     Additonally, the get_max_reward and get_min_reward methods should be overriden to return the maximum and minimum reward values respectively.
#     Optionally, the batch_call method can be overriden to compute the reward more efficiently in batch.
#     """
#     def __call__(
#         self,
#         model_output: Union[List[int], List[List[int]], torch.LongTensor],
#         ground_truth: Union[List[int], List[List[int]], torch.LongTensor],
#     ) -> Union[float, List[float], torch.Tensor]:
#         """ Call method for the reward function, this method should not be overriden. It checks the type of the input and calls the appropriate method. If the input is a batch of sequences, it calls the batch_call method, otherwise it calls the reward_fn method
        
#         :param model_output: Model output
#         :type model_output: Union[List[int], List[List[int]], torch.LongTensor]
#         :param ground_truth: Ground truth
#         :type ground_truth: Union[List[int], List[List[int]], torch.LongTensor
#         :return: Reward
#         :rtype: Union[float, List[float], torch.Tensor]    
        
#         """
#         assert isinstance(model_output, type(ground_truth)), "model_output and ground_truth must be of the same type"
        
#         # Type checking + determine if it is a batch call or not
#         is_batch_call = None
        
#         #Case 1: LongTensor
#         if isinstance(model_output, torch.LongTensor):
#             #if there's only one dimension, it is not a batch call
#             if len(model_output.shape) == 1:
#                 is_batch_call = False
#             #if there are two dimensions, it is a batch call
#             elif len(model_output.shape) == 2:
#                 is_batch_call = True                
        
#         #Case 2: List[int] or List[List[int]]
#         elif isinstance(model_output, list):
#             # if the first element is an int, it is not a batch call
#             if isinstance(model_output[0], int):
#                 is_batch_call = False
#             # if the first element is a list of ints, it is a batch call
#             elif isinstance(model_output[0], list) and isinstance(model_output[0][0], int):
#                 is_batch_call = True
                
#             #convert to torch tensor
#             model_output = torch.tensor(model_output)
#             ground_truth = torch.tensor(ground_truth)
        
#         #Case 3: Invalid type
#         if is_batch_call is None:
#             raise ValueError(
#                 "model_output and ground_truth must be either a LongTensor of shape (batch_size, seq_len) or (seq_len), \
#                     or a List[List[int]] or List[int]")
        
#         #call the appropriate method
#         if is_batch_call:
#             return self.batch_call(model_output, ground_truth)
#         else:
#             return self.reward_fn(model_output, ground_truth)
    
#     def batch_call(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor) -> List[float]:
#         """ Batch call method for the reward function, this method can be overriden by the subclass if the reward function can be computed more efficiently in batch (e.g. using tensor operations). Note: returning a tensor is permitted
        
#         :param model_output: Model output
#         :type model_output: torch.LongTensor
#         :param ground_truth: Ground truth
#         :type ground_truth: torch.LongTensor
#         :return: corresponding rewards
#         :rtype: Union[List[float], torch.Tensor]
#         """
        
#         return [
#                 self.reward_fn(model_output, ground_truth) 
#                 for model_output, ground_truth in zip(model_output, ground_truth)
#             ]
        
#     def reward_fn(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor):
#         """ Reward function, this method should be overriden by the subclass. It should return a float, the reward value.
        
#         :param model_output: Model output
#         :type model_output: torch.LongTensor
#         :param ground_truth: Ground truth
#         :type ground_truth: torch.LongTensor
#         :return: Reward
#         :rtype float
#         """
#         raise NotImplementedError
        
#     def get_max_reward(self):
#         """ This method should be overriden by the subclass. Get the maximum reward value
        
#         :return: Maximum reward value
#         :rtype: float
#         """
#         raise NotImplementedError
    
#     def get_min_reward(self):
#         """ This method should be overriden by the subclass. Get the minimum reward value
        
#         :return: Minimum reward value
#         :rtype: float
#         """
#         raise NotImplementedError
    
# class GSM8KCorrectnessReward(AbstractReward):
#     """ Reward function for the GSM8K dataset. This reward function checks if the model output is correct given the ground truth answer. The ground truth answer must be in the GSM8K dataset format
    
#     :param tokenizer: Tokenizer
#     :type tokenizer: transformers.PreTrainedTokenizer
#     """
    
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
        
#     def reward_fn(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor):
#         """An adaptation of thirdparty.openai.grade_school_math.dataset.is_correct. Reward function, returns 1.0 if the model output is correct w.r.t. ground truth, 0.0 if it is incorrect, -1.0 if the model output is invalid
        
#         :param model_output: Model output
#         :type model_output: torch.LongTensor
#         :param ground_truth: Ground truth
#         :type ground_truth: torch.LongTensor
#         :return: Reward
#         :rtype: float
#         """
#         #an adaptation of thirdparty.openai.grade_school_math.dataset.is_correct
        
#         #extract the answer of gt
#         gt_answer = extract_answer(strip_special_tokens(ground_truth,self.tokenizer))
#         assert gt_answer != INVALID_ANS, f"Ground truth answer is invalid, your ground truth answer is {ground_truth}"
#         #extract the answer of the model output
#         pred_answer = extract_answer(strip_special_tokens(model_output,self.tokenizer))
        
#         #if the model output is invalid, return -1.0
#         if pred_answer == INVALID_ANS:
#             return -1.0
#         #if the model output is correct, return 1.0 otherwise return 0.0
#         return float(pred_answer == gt_answer)
    
#     def get_max_reward(self):
#         """ Get the maximum reward value (1.0)
        
#         :return: Maximum reward value
#         :rtype: float
#         """
#         return 1.0
    
#     def get_min_reward(self):
#         """ Get the minimum reward value (-1.0)
        
#         :return: Minimum reward value
#         :rtype: float
#         """
#         return -1.0
    
# from gymnasium import Env
# from typing import List
# from transformers import PreTrainedTokenizer
# from datasets import Dataset

# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
# class LMDummyVecEnv(DummyVecEnv):
#     """ Vectorized environment for language model environments. This class is a subclass of DummyVecEnv and is used to handle observations of variable length. It is used to handle environments where the observation space is a sequence of tokens of variable length."""
#     def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
#         """ Save the observation in the buffer
        
#         :param env_idx: Environment index
#         :type env_idx: int
#         :param obs: Observation
#         :type obs: VecEnvObs
#         """
#         for key in self.keys:    
#             if key is None:
#                 #get the length of the observation and save it in the buffer
#                 len_obs = obs.shape[-1]
#                 self.buf_obs[key][env_idx][:len_obs] = obs
#             else:
#                 #get the length of the observation and save it in the buffer
#                 len_obs = obs[key].shape[-1]
#                 self.buf_obs[key][env_idx][:len_obs] = obs  # type: ignore[call-overload]
                
    
# from typing import Tuple, Any
# class LanguageModelEnv(Env):
#     """ Environment for language models. This class is a subclass of gym.Env and is used to handle language model environments. 
#     This environment allows to sample from a dataset and compute rewards based on the model output and the ground truth.
    
#     :param reward: Reward function used to compute the reward of observations
#     :type reward: AbstractReward
#     :param tokenizer: Tokenizer used to encode and decode text
#     :type tokenizer: PreTrainedTokenizer
#     :param termination_tokens: List of tokens that terminate the sequence
#     :type termination_tokens: List[int]
#     :param max_tokens: Maximum number of tokens in the observation
#     :type max_tokens: int
#     :param dataset: Dataset used to sample from
#     :type dataset: Dataset
#     :param filler_token: Filler token used to pad the observation
#     :type filler_token: int
#     """
#     # class variable pointer to dataset - it should be done only once
#     dataset = None 
#     def __init__(
#         self,
#         reward: AbstractReward,
#         tokenizer: PreTrainedTokenizer,
#         termination_tokens: List[int],
#         max_tokens: int,
#         dataset: Dataset = None,
#         filler_token: int = -100,
#     ):
#         super(LanguageModelEnv, self).__init__()

#         self.reward = reward
#         self.termination_tokens = termination_tokens
#         self.max_tokens = max_tokens
#         self.tokenizer = tokenizer
#         self.filler_token = filler_token
        
#         if not LanguageModelEnv.dataset:
#             if dataset is None:
#                 raise ValueError("dataset must be provided")
#             LanguageModelEnv.dataset = dataset
        
#         self.observation_space =  spaces.MultiDiscrete([tokenizer.vocab_size]* max_tokens, dtype = np.int64) 
#         self.action_space = spaces.Discrete(tokenizer.vocab_size)
#         self.current_state = []

#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
#         """ Apply an action to the environment. For a language model it's simply adding the action to the current state
        
#         :param action: Action to apply
#         :type action: int
#         :return: Observation, reward, termination signal, truncation signal, info
#         :rtype: Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
#         """
#         self.current_state.append(action.item())
#         observation , reward, terminated, truncated, info = self._get_obs()
#         return observation, reward, terminated, truncated, info

#     def is_terminated(self, state: List[int]):
#         """ Check if the state is terminated
        
#         :param state: State
#         :type state: List[int]
#         :return: True if the state is terminated, False otherwise
#         :rtype: bool
#         """
#         #skip the first token because it is the BOS token (which is sometimes the same as the EOS token)
#         return any([token in state[1:] for token in self.termination_tokens])
    
#     def is_truncated(self, state: List[int]):
#         """ Check if the state is truncated (i.e. the maximum number of tokens has been reached)
        
#         :param state: State
#         :type state: List[int]
#         :return: True if the state is truncated, False otherwise
#         :rtype: bool
#         """
#         if not self.is_terminated(state) and len(state) >= self.max_tokens:
#             return True
#         return False
    
#     def reset(
#         self,
#         seed = 123,
#         id: int = None,
#         options = None,
#     ):  # type: ignore
#         """ Reset the environment. This method samples a new example from the dataset and resets the environment
        
#         :param seed: Seed used to sample the example
#         :type seed: int
#         :param id: ID of the example to sample
#         :type id: int
#         :param options: Additional options
#         :type options: Any
#         :return: Observation and info
#         :rtype: Tuple[np.ndarray, Dict[str, Any]]
#         """
        
#         super().reset(seed=seed)
#         #sample a new example
#         if id is None:
#             id = self.np_random.choice(len(LanguageModelEnv.dataset))
#         input_sample = LanguageModelEnv.dataset[id]
#         input_text = input_sample["input_text"]
#         #save the output text (ground truth)
#         self.output_text = self.tokenizer(input_sample["output_text"], return_tensors="np", padding=True, truncation=True)["input_ids"].reshape(-1).tolist()
#         batch_encoding = self.tokenizer(input_text, return_tensors="np", padding=True, truncation=True)
#         #save the current state (input text)
#         self.current_state = batch_encoding["input_ids"].reshape(-1).tolist()
        
#         #return the observation and info
#         self.last_obs = self.current_state
#         self.terminated = False
#         self.truncated = False
#         self.done = False
#         return np.array(self.current_state), {} # return observation and info=None
    
#     def _get_obs(self):
#         """ Get the observation, reward, termination signal, truncation signal and info
        
#         :return: Observation, reward, termination signal, truncation signal, info
#         :rtype: Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
#         """
#         is_terminated =  self.is_terminated(self.current_state)
        
#         is_truncated = self.is_truncated(self.current_state)
#         reward = self.reward(self.current_state, self.output_text) if is_terminated or is_truncated else self.reward.get_min_reward()
        
#         info = {}
#         self.last_obs = self.current_state
#         full_current_state = self.current_state #self.resize_obs()
       
#         return np.array(full_current_state) , reward, is_terminated, is_truncated, info

#     def render(self):
#         """ Render the current state
        
#         :return: Current state
#         :rtype: str
#         """
#         return self.tokenizer.decode(self.current_state)

#     def close(self):
#         """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

#         This is critical for closing rendering windows, database or HTTP connections.
#         Calling ``close`` on an already closed environment has no effect and won't raise an error.
#         """
#         pass    
    
# from stable_baselines3.common.policies import BasePolicy
# from stable_baselines3.common.type_aliases import PyTorchObs
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
# from typing import Type, Optional, Dict, Any, Tuple
# from gymnasium import spaces
# import torch
# import gymnasium as gym
# from transformers import PreTrainedModel
# from transformers import PreTrainedTokenizer
# from stable_baselines3.common.type_aliases import Schedule
# from stable_baselines3.common.utils import get_linear_fn,obs_as_tensor
# class LLMBasePolicy(BasePolicy):
#     """ Base policy for language models. This class is a subclass of BasePolicy and is used to handle language model policies.
    
#     :param lr_schedule: Learning rate schedule
#     :type lr_schedule: Schedule
#     :param lm: Language model
#     :type lm: PreTrainedModel
#     :param tokenizer: Tokenizer
#     :type tokenizer: PreTrainedTokenizer
#     :param observation_space: Observation space (if None, it is set to a MultiDiscrete space with the vocab size of the language model)
#     :type observation_space: spaces.Space
#     :param action_space: Action space (if None, it is set to a Discrete space with the vocab size of the language model)
#     :type action_space: spaces.Space
#     :param features_extractor_class: Features extractor class
#     :type features_extractor_class: Type[BaseFeaturesExtractor]
#     :param features_extractor_kwargs: Features extractor keyword arguments
#     :type features_extractor_kwargs: Optional[Dict[str, Any]]
#     :param normalize_images: Whether to normalize images
#     :type normalize_images: bool
#     :param optimizer_class: Optimizer class
#     :type optimizer_class: torch.optim.Optimizer
#     :param optimizer_kwargs: Optimizer keyword arguments
#     :type optimizer_kwargs: Optional[Dict[str, Any]]
#     :param squash_output: Whether to squash the output
#     :type squash_output: bool
#     :param filler_token: Filler token
#     :type filler_token: int
#     """
    
    
#     def __init__(
#         self,
#         observation_space: spaces.Space = None,
#         action_space: spaces.Space = None,
#         lr_schedule: Schedule = None,
#         tokenizer: PreTrainedTokenizer = None,
#         lm: PreTrainedModel = None,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = False,
#         optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#         squash_output: bool = False,
#         filler_token: int = -100,
#         **kwargs
#     ):
#         #check if observation_space and action_space are provided and set them to the default values if they are not
#         if observation_space is None:
#             #check vocab size
#             observation_space = spaces.MultiDiscrete([lm.config.vocab_size]* tokenizer.vocab_size, dtype = np.int64) 
#         if action_space is None:
#             action_space = spaces.Discrete(lm.config.vocab_size)
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor_class,
#             features_extractor_kwargs,
#             optimizer_class=optimizer_class,
#             optimizer_kwargs=optimizer_kwargs,
#             squash_output=squash_output,
#             normalize_images=normalize_images,
#         )
#         self.lm = lm
#         self.tokenizer = tokenizer
#         self.filler_token = filler_token
#         self._build(lr_schedule)
        
#     def _build(self, lr_schedule: Schedule):
#         """ Build the policy and optimizer
        
#         :param lr_schedule: Learning rate schedule
#         :type lr_schedule: Schedule
#         """
#         self.optimizer = self.optimizer_class(
#             self.lm.parameters(),
#             lr=lr_schedule(1),  # type: ignore[call-arg]
#             **self.optimizer_kwargs,
#         )
#         self.lm.eval()
    
#     def obs_to_tensor(self, observation: np.ndarray) -> Tuple[PyTorchObs, bool]:    
#         """ Convert an observation to a PyTorch tensor
        
#         :param observation: Observation
#         :type observation: np.ndarray
#         :return: Observation tensor and whether the observation is valid
#         :rtype: Tuple[PyTorchObs, bool]
#         """
#         assert isinstance(observation, np.ndarray), "Observation must be a numpy array"
#         obs_tensor = obs_as_tensor(observation, self.device)
#         return obs_tensor, True
        
#     def compute_nll_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         """ Compute the negative log likelihood loss
        
#         :param logits: Logits
#         :type logits: torch.Tensor
#         :param labels: Labels
#         :type labels: torch.Tensor
#         :return: Negative log likelihood loss
#         :rtype: torch.Tensor
#         """
#         shift_lm_logits = logits[..., :-1, :].contiguous()
#         shift_lm_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         shift_lm_logits = shift_lm_logits.view(-1, self.lm.config.vocab_size)
#         shift_lm_labels = shift_lm_labels.view(-1)
#         # Ensure tensors are on the same device
#         shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)
#         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.filler_token)
#         return loss_fct(shift_lm_logits, shift_lm_labels)
    
    
#     def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PyTorchObs:
#         if (isinstance(obs, dict) and 'input_ids' in obs and 'attention_mask' not in obs) or isinstance(obs, torch.Tensor):
#             warnings.warn("Attention mask not provided, the padding mask will be automatically computed")
#             obs_to_pass = obs if isinstance(obs, torch.Tensor) else obs["input_ids"]
#             feature = self.tokenizer.pad({"input_ids": obs_to_pass}, return_tensors="pt", padding=True)
#         elif isinstance(obs, dict) and "input_ids" in obs and "attention_mask" in obs:
#             feature = obs
#         else:
#             raise ValueError("Observation type not supported")
#         return feature
            
            
#     def forward(self, obs: PyTorchObs, action = None) -> torch.Tensor:
        
#         feature = self.extract_features(obs)
#         feature = {k: v.to(self.device) for k, v in feature.items()}
#         return self.lm(**feature)
    
#     def _predict(self, observation: PyTorchObs, deterministic: bool = False):
#         """
#         Get the action according to the policy for a given observation.

#         By default provides a dummy implementation -- not all BasePolicy classes
#         implement this, e.g. if they are a Critic in an Actor-Critic method.

#         :param observation:
#         :param deterministic: Whether to use stochastic or deterministic actions
#         :return: Taken action according to the policy
#         """
#         output = self.forward(observation)
#         logits = output.logits[...,-1,:]
#         probs = torch.nn.functional.softmax(logits, dim=-1)
#         if deterministic:
#             return probs.argmax(dim=-1)
#         else:
#             return torch.multinomial(probs, num_samples=1)

# from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
# from stable_baselines3.common.vec_env import VecNormalize 
# from stable_baselines3.common.type_aliases import ReplayBufferSamples
# from stable_baselines3.common.buffers import ReplayBuffer
# import warnings
# import transformers
# class LMReplayBuffer(ReplayBuffer):
#     def __init__(
#         self,
#         *args,
#         tokenizer,
#         reward_threshold: float = None,
#         filler_token = -100, 
#         **kwargs
#     ):
#         if "n_envs" in kwargs and kwargs["n_envs"] > 1:
#             warnings.warn("LMReplayBuffer does not distinguish between environments, n_envs will be ignored and set to 1")
#             kwargs["n_envs"] = 1
#         super().__init__(*args, **kwargs)
#         self.filler_token = filler_token
#         self.observations.fill(self.filler_token)
#         self.actions.fill(self.filler_token)
#         if self.optimize_memory_usage:
#             self.next_observations.fill(self.filler_token)
#         self.tokenizer = tokenizer
#         self.reward_threshold = reward_threshold
        
#     def set_filler_token(self, filler_token):
        
#         self.observations[self.observations == self.filler_token] = filler_token
#         self.actions[self.actions == self.filler_token] = filler_token
#         if self.optimize_memory_usage:
#             self.next_observations[self.next_observations == self.filler_token] = filler_token
#         self.filler_token = filler_token
    
#     def to_torch(self, array: Union[np.ndarray, torch.Tensor, transformers.BatchEncoding], copy: bool = True) -> Union[torch.Tensor, transformers.BatchEncoding]:
#         if isinstance(array, transformers.BatchEncoding):
#             return {k: v.to(self.device) for k,v in array.items()}
#         elif isinstance(array, torch.Tensor):
#             return array.to(self.device)
#         return super().to_torch(array, copy)
    
#     def add(
#         self,
#         obs: np.ndarray,
#         next_obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         #find sequences where done is True
#         done_indices = np.where(done)[0]
#         finished_rewards = reward[done_indices]
#         #check if rewards are above threshold
#         if self.reward_threshold is not None:
#             above_thrsh_rewards = np.where(finished_rewards > self.reward_threshold)[0]
#         else:
#             above_thrsh_rewards = None
            
#         if len(done_indices) > 0 and (above_thrsh_rewards is None or len(above_thrsh_rewards) > 0):
#             finished_rewards = finished_rewards[above_thrsh_rewards]
#             finished_obs = obs[done_indices][above_thrsh_rewards]
#             finished_actions = action[done_indices][above_thrsh_rewards]
            
#             finished_infos = [infos[i] for i in done_indices]
#             if above_thrsh_rewards is not None:
#                 finished_infos = [finished_infos[i] for i in above_thrsh_rewards]
#             finished_next_obs = next_obs[done_indices][above_thrsh_rewards]
#             finished_done = done[done_indices][above_thrsh_rewards]
            
#             if (above_thrsh_rewards is not None and len(above_thrsh_rewards) > 1) or above_thrsh_rewards is None and len(done_indices) > 1:
#                 super().add(finished_obs, finished_next_obs, finished_actions, finished_rewards, finished_done, finished_infos)
#             for i in range(len(finished_obs)):
#                 super().add(finished_obs[i], finished_next_obs[i], finished_actions[i], finished_rewards[i], finished_done[i], finished_infos[i])
            
#     def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
#         # Sample randomly the env idx
#         env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

#         if self.optimize_memory_usage:
#             next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
#         else:
#             next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

#         next_obs = self.tokenizer(
#             self.tokenizer.batch_decode(
#                 remove_filler_tokens(next_obs, self.filler_token)
#             ),
#             return_tensors="pt", padding=True, truncation=True
#         )
        
#         obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        
#         obs = self.tokenizer(
#             self.tokenizer.batch_decode(
#                 remove_filler_tokens(self.observations[batch_inds, env_indices, :], self.filler_token)
#             ),
#             return_tensors="pt", padding=True, truncation=True
#         )

#         actions = self.tokenizer(
#             self.tokenizer.batch_decode(
#                 remove_filler_tokens(self.actions[batch_inds, env_indices, :], self.filler_token)
#             , return_tensors="pt", padding=True, truncation=True)
#         )["input_ids"]
    
#         data = (
#             obs,
#             actions,
#             next_obs,
#             # Only use dones that are not due to timeouts
#             # deactivated by default (timeouts is initialized as an array of False)
#             (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
#             self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
#         )
#         return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

# from stable_baselines3.common.vec_env import VecEnv
# from stable_baselines3.common.noise import ActionNoise
# from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# class STaR(OffPolicyAlgorithm):
    
#     def __init__(self,*args,**kwargs):
#         kwargs["support_multi_env"] = True
#         super().__init__(*args, **kwargs)
#         self._setup_model()
#         assert all([isinstance(myenv, LanguageModelEnv) for myenv in self.env.envs]), "All environments must be of type LanguageModelEnv"
#         all_filler_token = [myenv.filler_token for myenv in self.env.envs]
#         assert all([filler_token == all_filler_token[0] for filler_token in all_filler_token]), "All environments must have the same filler token"
#         self.policy.filler_token = all_filler_token[0]
#         self.replay_buffer.set_filler_token(all_filler_token[0])
        
        
#     def collect_rollouts(
#         self,
#         env: VecEnv,
#         callback: BaseCallback,
#         train_freq: TrainFreq,
#         replay_buffer: ReplayBuffer,
#         action_noise: Optional[ActionNoise] = None,
#         learning_starts: int = 0,
#         log_interval: Optional[int] = None,
#     ) -> RolloutReturn:
        
#         og_padding_side = self.policy.tokenizer.padding_side
#         self.policy.tokenizer.padding_side = "left"
#         res = super().collect_rollouts(
#             env,
#             callback,
#             train_freq,
#             replay_buffer,
#             action_noise,
#             learning_starts,
#             log_interval
#         )
#         self.policy.tokenizer.padding_side = og_padding_side
#         return res
    
#     def _store_transition(
#         self,
#         replay_buffer: ReplayBuffer,
#         buffer_action: np.ndarray,
#         new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
#         reward: np.ndarray,
#         dones: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         """
#         Store transition in the replay buffer.
#         We store the normalized action and the unnormalized observation.
#         It also handles terminal observations (because VecEnv resets automatically).

#         :param replay_buffer: Replay buffer object where to store the transition.
#         :param buffer_action: normalized action
#         :param new_obs: next observation in the current episode
#             or first observation of the episode (when dones is True)
#         :param reward: reward for the current transition
#         :param dones: Termination signal
#         :param infos: List of additional information about the transition.
#             It may contain the terminal observations and information about timeout.
#         """
#         for i, done in enumerate(dones):
#             if done and infos[i].get("terminal_observation") is not None:
#                 infos[i]["terminal_observation"] = \
#                     add_filler_tokens(
#                         infos[i]["terminal_observation"],
#                         len(self.observation_space),
#                         self.policy.filler_token
#                     )
#         super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
                        
    
#     def train(self, gradient_steps: int, batch_size: int) -> None:
#         self.policy.train()
        
#         self._update_learning_rate(self.policy.optimizer)
                
#         nll_losses = []
        
#         for _ in range(gradient_steps):
#             self._n_updates += 1
            
#             replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

#             output = self.policy(replay_data.observations)
            
#             nll_loss = self.policy.compute_nll_loss(output.logits, replay_data.observations)
            
#             nll_losses.append(nll_loss.item())
            
#             self.policy.optimizer.zero_grad()
            
#             nll_loss.backward()
            
#             # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
#             self.policy.optimizer.step()
                    
            
#         self.logger.record("train/nll_loss", np.mean(nll_losses))
#         self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        
# from stable_baselines3.common.callbacks import BaseCallback
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from datasets import Dataset, load_dataset

from lm_stable_baselines.environments import LanguageModelEnv
from lm_stable_baselines.environments.vectorized_environments import LMDummyVecEnv
from lm_stable_baselines.rewards import GSM8KCorrectnessReward
from lm_stable_baselines.buffers import LMReplayBuffer
from lm_stable_baselines.training_algorithms import STaR
from lm_stable_baselines.policies import LLMBasePolicy

def create_env(reward,tokenizer, eos_token_id, max_tok, dataset, filler_token):
    return LanguageModelEnv(reward,tokenizer, [eos_token_id], max_tok, dataset, filler_token=filler_token)


if __name__ == "__main__":
    #load gpt2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map = "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.unk_token
    #get eos token id
    eos_token_id = tokenizer.eos_token_id
    #initialize reward
    
    reward = GSM8KCorrectnessReward(tokenizer)
    input_dir = "src/data/gsm8k_json/gsm8k/"
    dataset = load_dataset('json', data_files=input_dir + 'train.json', split='train', )
    dataset = dataset.rename_column("question", "input_text").rename_column("answer", "output_text")
    
    env = LMDummyVecEnv(
        [
            lambda: create_env(reward,tokenizer, eos_token_id, 1024, dataset, filler_token=tokenizer.pad_token_id),
            lambda: create_env(reward,tokenizer, eos_token_id, 1024, dataset, filler_token=tokenizer.pad_token_id),
        ]
    )
    policy = LLMBasePolicy
    policy_kwargs = {"lm": model, "tokenizer": tokenizer}
    
    algo = STaR(
        policy = policy,
        policy_kwargs = policy_kwargs,
        env=env,
        learning_rate=1e-5,
        train_freq = TrainFreq(1000, TrainFrequencyUnit.STEP),
        replay_buffer_class= LMReplayBuffer,
        replay_buffer_kwargs={"tokenizer": tokenizer, "reward_threshold": reward.get_min_reward()},
        batch_size = 8,
    )
    #Note: You will get the error
    #  File "numpy/random/_bounded_integers.pyx", line 1334, in numpy.random._bounded_integers._rand_int64
    # ValueError: high <= 0
    # this will fail because gpt2 is doesn't generate good enough rewards and the "reward_threshold" argument filters on rewards
    algo.learn(100)
