from typing import Tuple, Any
from gymnasium import Env,spaces
from typing import List, Dict
from transformers import PreTrainedTokenizer
from datasets import Dataset
from lm_stable_baselines.rewards import AbstractReward
import numpy as np
from lm_stable_baselines.utils import remove_filler_tokens
import warnings
import torch
from torch import LongTensor
from src.utils.constants import ANSWER_TEMPLATE
class LanguageModelEnv(Env):
    """ Environment for language models. This class is a subclass of gym.Env and is used to handle language model environments. 
    This environment allows to sample from a dataset and compute rewards based on the model output and the ground truth.
    
    :param reward: Reward function used to compute the reward of observations
    :type reward: AbstractReward
    :param tokenizer: Tokenizer used to encode and decode text
    :type tokenizer: PreTrainedTokenizer
    :param termination_tokens: List of tokens that terminate the sequence
    :type termination_tokens: List[int]
    :param max_tokens: Maximum number of tokens in the observation
    :type max_tokens: int
    :param dataset: Dataset used to sample from
    :type dataset: Dataset
    :param filler_token: Filler token used to pad the observation
    :type filler_token: int
    """
    # class variable pointer to dataset - it should be done only once
    dataset: Dataset = None
    stage: str = "train"
    next_idx: int = 0
    read_sequentially: bool = False
    gt_array: np.ndarray = None
    last_gt_pos: int = 0
    idx_of_last_in_context_gt_reasoning_step_distr = None
    
    
    def __init__(
        self,
        reward: AbstractReward,
        tokenizer: PreTrainedTokenizer,
        termination_tokens: List[int],
        max_tokens: int,
        dataset: Dataset = None,
        require_dataset: bool = False,
        filler_token: int = -100,
        n_envs = -1,
        env_idx = -1,
        enable_delta_reward = False,
        max_actions=1,
        reasoning_step_splitter = ' ', # could be '\n' for example
        ground_truth_portion = 0, 
    ):
        super(LanguageModelEnv, self).__init__()

        self.reward = reward
        self.termination_tokens = termination_tokens
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.filler_token = filler_token
        self.require_dataset = require_dataset
        self.max_actions = max_actions

        LanguageModelEnv.n_envs = n_envs
        self.env_idx = env_idx

        self.gt_id = LanguageModelEnv.last_gt_pos
        LanguageModelEnv.last_gt_pos += 1

        if LanguageModelEnv.gt_array is None:
            #initialize gt_array of dimention n_envs, max_tokens filled with -100
            LanguageModelEnv.gt_array = np.full((n_envs, max_tokens), -100)

        if require_dataset and not LanguageModelEnv.dataset:
            if dataset is None:
                raise ValueError("dataset must be provided")
            LanguageModelEnv.dataset = dataset
            LanguageModelEnv.reprermute_dataset_id_list()

        self.observation_space =  spaces.MultiDiscrete([tokenizer.vocab_size]* max_tokens, dtype = np.int64) 
        self.action_space = spaces.MultiDiscrete([tokenizer.vocab_size]* max_tokens, dtype = np.int64)
        self.current_state = []
        self.enable_delta_reward = enable_delta_reward
        self.n_actions_taken = 0

        self.reasoning_step_splitter = reasoning_step_splitter
        self.ground_truth_portion = ground_truth_portion if ground_truth_portion is not None else 0 #portion of the ground truth actions that are given to the agent, the rest should be predicted by 
        self.ground_truth_portions = []

    @classmethod
    def reprermute_dataset_id_list(cls):
        if cls.read_sequentially:
            cls.dataset_id_list = list(range(len(LanguageModelEnv.dataset[cls.stage])))
        else:
            cls.dataset_id_list = np.random.permutation(len(LanguageModelEnv.dataset[cls.stage]))
        cls.next_idx = 0
        # TODO: check if this is necessary
        # NICKY: 
        #   I don't think we nee this. We want dataset_id_list to be a static variable that is shared across all instances of the class
        #   We know which sample to take thanks to LanguageModelEnv.next_idx
        # if LanguageModelEnv.n_envs != -1:
        #     self.dataset_id_list = LanguageModelEnv.dataset_id_list[self.env_idx::LanguageModelEnv.n_envs]
        # else:
        #     self.dataset_id_list = LanguageModelEnv.dataset_id_list

    def _step(self, curr_obs, action):
        if isinstance(curr_obs, list):
            curr_obs.extend(action)
        elif isinstance(curr_obs, torch.Tensor):
            curr_obs = torch.cat([curr_obs, action], dim = 0)
        else:
            raise ValueError("curr_obs should be a list or a tensor")
        return curr_obs


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """ Apply an action to the environment. For a language model it's simply adding the action to the current state
        
        :param action: Action to apply
        :type action: int
        :return: Observation, reward, termination signal, truncation signal, info
        :rtype: Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
        """
        self.n_actions_taken += 1
        clean_action = remove_filler_tokens(action, self.filler_token).squeeze(-1).tolist()
        self.current_state = self._step(self.current_state, clean_action)
        observation , reward, terminated, truncated, info = self._get_obs()
        if self.enable_delta_reward:
            reward = reward - self.last_reward
            self.last_reward = reward
        
        return observation, reward, terminated, truncated, info
    
    def next_observation_from_observation_and_action(self, obs: LongTensor, actions: LongTensor) -> List[List[int]]:
        #assumption: filler tokens have been removed
        unpadded_obs = remove_filler_tokens(obs, self.tokenizer.pad_token_id)
        unpadded_acts = remove_filler_tokens(actions, self.tokenizer.pad_token_id)

        new_observations = [self._step(observation,action) for observation, action in zip(unpadded_obs,unpadded_acts)]

        return new_observations
    
    
    def is_terminated(self, state: List[int]):
        """ Check if the state is terminated
        
        :param state: State
        :type state: List[int]
        :return: True if the state is terminated, False otherwise
        :rtype: bool
        """
        #skip the first token because it is the BOS token (which is sometimes the same as the EOS token)
        return any([token in state[1:] for token in self.termination_tokens])
    
    def is_truncated(self, state: List[int]):
        """ Check if the state is truncated (i.e. the maximum number of tokens has been reached)
        
        :param state: State
        :type state: List[int]
        :return: True if the state is truncated, False otherwise
        :rtype: bool
        """

        if not self.is_terminated(state):
            reached_max_tokens = len(state) >= self.max_tokens
            reached_max_actions = self.n_actions_taken >= self.max_actions
            return reached_max_actions or reached_max_tokens
        return False
    
    @classmethod
    def set_stage(cls, stage: str, read_sequentially: bool = False):
        valid_stages = ["train", "val", "test"]
        assert stage in valid_stages, f"stage must be one of {valid_stages}"
        assert stage in LanguageModelEnv.dataset, f"stage {stage} not found in dataset"
        cls.stage = stage
        cls.next_idx = 0
        cls.read_sequentially = read_sequentially
        cls.reprermute_dataset_id_list()
        
    @classmethod
    def get_ground_truths(cls, stage: str, idxs: List[int]):
        """ Get the ground truths for the given stage and indices
        
        :param stage: Stage
        :type stage: str
        :param idxs: Indices
        :type idxs: List[int]
        :return: Ground truths
        :rtype: List[str]
        """
        return [cls.dataset[stage]["output"][idx] for idx in idxs]
        
    def reset(
        self,
        seed = 123,
        id = None,
        options = None,
    ):  # type: ignore
        """ Reset the environment. This method samples a new example from the dataset and resets the environment
        
        :param seed: Seed used to sample the example
        :type seed: int
        :param id: ID of the example to sample
        :type id: int
        :param options: Additional options
        :type options: Any
        :return: Observation and info
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        
        super().reset(seed=seed)
        #sample a new example
        ground_truth_portion = self.sample_portion()
        self.ground_truth_portions.append(ground_truth_portion)
        if self.require_dataset:
            #if we reached the end of the dataset, repermute the dataset
            if LanguageModelEnv.next_idx >= len(self.dataset_id_list):
                LanguageModelEnv.reprermute_dataset_id_list()
            #sample the next example (safe because if its the last one, we will have repermuted the dataset)
            idx = LanguageModelEnv.next_idx
            id = int(self.dataset_id_list[idx])
            LanguageModelEnv.next_idx = idx + 1
        else:
            raise ValueError("not implemented without dataset yet")

        input_sample = LanguageModelEnv.dataset[LanguageModelEnv.stage][id]
        input_text = input_sample["input"]

        #  add self.ground_truth_portion of the chain of thought tokens (or actions if self.reasoning_step_splitter is not None) to the observation
        if LanguageModelEnv.stage == "train":
            if ANSWER_TEMPLATE in input_sample["output"]:
                #keep only the reasoning steps after ANSWER_TEMPLATE
                reasoning_steps = input_sample["output"].split(ANSWER_TEMPLATE)[1]
            else:
                reasoning_steps = input_sample["output"]
            
            reasoning_steps = reasoning_steps.split(self.reasoning_step_splitter)
            reasoning_length = len(reasoning_steps)
            supervised_length = int(ground_truth_portion*reasoning_length)
            reasoning_steps = self.reasoning_step_splitter.join(reasoning_steps[:supervised_length])
            input_text = input_text + reasoning_steps + self.reasoning_step_splitter

        #save the output text (ground truth)
        self.output_text = self.tokenizer(input_sample["output"], return_tensors="np", padding=True, truncation=True)["input_ids"].reshape(-1).tolist()
        
        batch_encoding = self.tokenizer(input_text, return_tensors="np", padding=True, truncation=True)
        #save the current state (input text)
        self.current_state = batch_encoding["input_ids"].reshape(-1).tolist()
        self.n_actions_taken = 0
        if len(self.current_state) > self.max_tokens:
            warnings.warn(f"The sampled input text here below is longer than max_tokens ({len(self.current_state)} > {self.max_tokens}): \n {self.input_text} \n Another example will be sampled")
            return self.reset(seed=seed, options=options)
        
        #return the observation and info
        self.terminated = False
        self.truncated = False
        self.done = False

        LanguageModelEnv.gt_array[ self.gt_id , :len(self.output_text)] = self.output_text
        LanguageModelEnv.gt_array[ self.gt_id , len(self.output_text):] = -100
        
        if self.enable_delta_reward:
            self.last_reward = self.reward(self.current_state, self.output_text)
        
        return np.array(self.current_state), {} # return observation and info=None
    
    def _get_obs(self):
        """ Get the observation, reward, termination signal, truncation signal and info
        
        :return: Observation, reward, termination signal, truncation signal, info
        :rtype: Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
        """
        is_terminated =  self.is_terminated(self.current_state)
        
        is_truncated = self.is_truncated(self.current_state)
        reward = self.reward(self.current_state, self.output_text)
        
        info = {}
        return np.array(self.current_state) , reward, is_terminated, is_truncated, info

    def render(self):
        """ Render the current state
        
        :return: Current state
        :rtype: str
        """
        return self.tokenizer.decode(self.current_state)

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        pass   

    def set_portion(self, portion):
        self.ground_truth_portion_dist = portion
        self.ground_truth_portions = []

    def sample_portion(self):
        if callable(self.ground_truth_portion_dist):
            return self.ground_truth_portion_dist(size=1)
        elif isinstance(self.ground_truth_portion_dist, float):
            return self.ground_truth_portion_dist
        else:
            raise ValueError("ground_truth_portion_dist should be a float or a callable")
        
    def compute_portion_from_obs_actions(self, rollout_data) -> float:
        #assumption: filler tokens have been removed
        obs = rollout_data.observations["input_ids"]
        actions = rollout_data.actions

        obs = remove_filler_tokens(obs, self.tokenizer.pad_token_id)
        actions = remove_filler_tokens(actions, self.tokenizer.pad_token_id)

        obs = [self.tokenizer.decode(o, skip_special_tokens=True) for o in obs]
        actions = [self.tokenizer.decode(a, skip_special_tokens=True) for a in actions]

        action_steps = [a.split(self.reasoning_step_splitter) for a in actions]

        observed_ratio = np.zeros(len(obs))
        for i, ob in enumerate(obs):
            if ANSWER_TEMPLATE in ob:
                #keep only the reasoning steps after ANSWER_TEMPLATE
                reasoning_steps_in_obs = (ob.split(ANSWER_TEMPLATE)[1]).split(self.reasoning_step_splitter)
            else:
                reasoning_steps_in_obs = ob.split(self.reasoning_step_splitter)

            ratio = len(reasoning_steps_in_obs) / (len(action_steps[i]) + len(reasoning_steps_in_obs))
            
            observed_ratio[i] = ratio
        
        return ratio