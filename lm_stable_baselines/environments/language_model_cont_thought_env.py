from typing import Tuple, Any, Dict, List, Union
from gymnasium import Env, spaces
from transformers import PreTrainedTokenizer
from datasets import Dataset
from lm_stable_baselines.rewards import AbstractReward
import numpy as np
from lm_stable_baselines.utils import remove_filler_tokens
import warnings
import torch
from torch import LongTensor, FloatTensor, Tensor
from src.utils.constants import ANSWER_TEMPLATE

class LanguageModelContThoughtEnv(Env):
    """Environment for language models with continuous hidden state outputs.
    This environment allows sampling from a dataset and computing rewards based on both 
    discrete token outputs and continuous hidden state representations.
    
    Args:
        reward (AbstractReward): Reward function for observations
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding/decoding text
        termination_tokens (List[int]): Tokens that terminate sequences
        max_tokens (int): Maximum tokens in observation
        hidden_size (int): Size of hidden state vectors
        dataset (Dataset, optional): Dataset to sample from
        filler_token (int, optional): Token used for padding. Defaults to -100
        hidden_dtype (torch.dtype, optional): Data type for hidden states. Defaults to torch.float32
    """
    
    dataset: Dataset = None
    stage: str = "train"
    next_idx: int = 0
    read_sequentially: bool = False
    gt_array: np.ndarray = None
    last_gt_pos: int = 0
    n_rollouts_per_sample = 1
    
    def __init__(
        self,
        reward: AbstractReward,
        tokenizer: PreTrainedTokenizer,
        termination_tokens: List[int],
        max_tokens: int,
        hidden_size: int,
        dataset: Dataset = None,
        require_dataset: bool = False,
        filler_token: int = -100,
        n_envs = -1,
        env_idx = -1,
        enable_delta_reward = False,
        max_actions = 1,
        hidden_dtype = torch.float32,
        n_rollouts_per_sample = 1,
        reasoning_step_splitter = ' ', # could be '\n' for example
        ground_truth_portion_dist = 0, 
        ft_on_action_only = False,
    ):
        super().__init__()

        self.reward = reward
        self.termination_tokens = termination_tokens
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.filler_token = filler_token
        self.require_dataset = require_dataset
        self.max_actions = max_actions
        self.hidden_dtype = hidden_dtype
        self.reasoning_step_splitter = reasoning_step_splitter
        self.ground_truth_portion_dist = ground_truth_portion_dist #portion of the ground truth actions that are given to the agent, the rest should be predicted by 
        self.ft_on_action_only = ft_on_action_only

        LanguageModelContThoughtEnv.n_envs = n_envs
        self.env_idx = env_idx
        
        self.gt_id = LanguageModelContThoughtEnv.last_gt_pos
        LanguageModelContThoughtEnv.last_gt_pos += 1

        if LanguageModelContThoughtEnv.gt_array is None:
            LanguageModelContThoughtEnv.gt_array = np.full((n_envs, max_tokens), -100)

        if require_dataset and not LanguageModelContThoughtEnv.dataset:
            if dataset is None:
                raise ValueError("dataset must be provided")
            LanguageModelContThoughtEnv.dataset = dataset
            LanguageModelContThoughtEnv.reprermute_dataset_id_list()

        # Define observation and action spaces for both discrete tokens and continuous hidden states
        self.observation_space = spaces.Dict({
            'input_ids': spaces.MultiDiscrete([tokenizer.vocab_size] * max_tokens),
            'hidden_states': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(max_tokens, hidden_size),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Dict({
            'input_ids': spaces.MultiDiscrete([tokenizer.vocab_size] * max_tokens),
            'hidden_states': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_tokens, hidden_size),
                dtype=np.float32
            )
        })

        self.current_state = {
            'input_ids': [],
            'hidden_states': []
        }
        
        self.enable_delta_reward = enable_delta_reward
        self.n_actions_taken = 0
        LanguageModelContThoughtEnv.n_rollouts_per_sample = n_rollouts_per_sample

    @classmethod
    def reprermute_dataset_id_list(cls):
        if cls.read_sequentially:
            cls.dataset_id_list = list(range(len(cls.dataset[cls.stage])))
        else:
            cls.dataset_id_list = np.random.permutation(len(cls.dataset[cls.stage]))
        cls.next_idx = 0
        
        if cls.stage == "train":
            new_dataset_id_list = []
            for item in cls.dataset_id_list:
                for _ in range(cls.n_rollouts_per_sample):
                    new_dataset_id_list.append(item)
            cls.dataset_id_list = new_dataset_id_list

    def _step(self, curr_obs: Dict[str, Union[List, torch.Tensor]], action: Dict[str, torch.Tensor]) -> Dict[str, Union[List, torch.Tensor]]:
        """Update current observation with new action"""
        if isinstance(curr_obs['input_ids'], list):
            curr_obs['input_ids'].extend(action['input_ids'])
            curr_obs['hidden_states'].extend(action['hidden_states'])
        elif isinstance(curr_obs['input_ids'], torch.tensor):
            curr_obs['input_ids'] = torch.cat([curr_obs['input_ids'], action['input_ids']], dim=0)
            curr_obs['hidden_states'] = torch.cat([curr_obs['hidden_states'], action['hidden_states']], dim=0)
        else:
            raise NotImplementedError
        return curr_obs

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment with both discrete and continuous actions"""
        self.n_actions_taken += 1
        
        # Clean and process actions
        clean_action = {
            'input_ids': remove_filler_tokens(action['input_ids'], self.filler_token).squeeze(-1).tolist(),
            'hidden_states': action['hidden_states'][action['input_ids'] != self.filler_token]
        }
        
        self.current_state = self._step(self.current_state, clean_action)
        observation, reward, terminated, truncated, info = self._get_obs()
        
        if self.enable_delta_reward:
            reward = reward - self.last_reward
            self.last_reward = reward
            
        return observation, reward, terminated, truncated, info

    def is_terminated(self, state: Dict[str, Union[List[int], np.ndarray]]) -> bool:
        """Check if sequence is terminated"""
        return any(token in state['input_ids'][1:] for token in self.termination_tokens)
    
    def is_truncated(self, state: Dict[str, Union[List[int], np.ndarray]]) -> bool:
        """Check if sequence is truncated"""
        if not self.is_terminated(state):
            reached_max_tokens = len(state['input_ids']) >= self.max_tokens
            reached_max_actions = self.n_actions_taken >= self.max_actions
            return reached_max_actions or reached_max_tokens
        return False

    @classmethod
    def set_stage(cls, stage: str, read_sequentially: bool = False):
        valid_stages = ["train", "val", "test"]
        assert stage in valid_stages, f"stage must be one of {valid_stages}"
        assert stage in cls.dataset, f"stage {stage} not found in dataset"
        cls.stage = stage
        cls.next_idx = 0
        cls.read_sequentially = read_sequentially
        cls.reprermute_dataset_id_list()

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observation"""
        super().reset(seed=seed)
        
        if not self.require_dataset:
            raise ValueError("Dataset required for this environment")
            
        if LanguageModelContThoughtEnv.next_idx >= len(self.dataset_id_list):
            LanguageModelContThoughtEnv.reprermute_dataset_id_list()
            
        idx = LanguageModelContThoughtEnv.next_idx
        id = int(self.dataset_id_list[idx])
        LanguageModelContThoughtEnv.next_idx = idx + 1

        input_sample = self.dataset[self.stage][id]
        input_text = input_sample["input"]

        # cut at portions:
        ground_truth_portion = self.sample_portion()
        if LanguageModelContThoughtEnv.stage == "train" and (idx % LanguageModelContThoughtEnv.n_rollouts_per_sample != 0  or LanguageModelContThoughtEnv.n_rollouts_per_sample == 1):
            if ANSWER_TEMPLATE in input_sample["output"]:
                #keep only the reasoning steps after ANSWER_TEMPLATE
                reasoning_steps = input_sample["output"].split(ANSWER_TEMPLATE)[1]
            else:
                reasoning_steps = input_sample["output"]
            
            reasoning_steps = reasoning_steps.split(self.reasoning_step_splitter)
            reasoning_length = len(reasoning_steps)
            supervised_length = int(ground_truth_portion*reasoning_length)
            if supervised_length == reasoning_length and self.ft_on_action_only:
                supervised_length -= 1
            reasoning_steps = self.reasoning_step_splitter.join(reasoning_steps[:supervised_length])
            input_text = input_text + reasoning_steps + self.reasoning_step_splitter

        
        # Encode input text
        batch_encoding = self.tokenizer(
            input_text, 
            return_tensors="np", 
            padding=True, 
            truncation=True
        )
        
        # Initialize hidden states with zeros
        hidden_states = np.zeros((len(batch_encoding["input_ids"][0]), self.hidden_size), dtype=np.float32)
        
        self.current_state = {
            'input_ids': batch_encoding["input_ids"].reshape(-1).tolist(),
            'hidden_states': hidden_states
        }
        
        self.output_text = input_sample["output"]
        self.n_actions_taken = 0

        if len(self.current_state['input_ids']) > self.max_tokens:
            warnings.warn(f"Input text too long ({len(self.current_state['input_ids'])} > {self.max_tokens})")
            return self.reset(seed=seed, options=options)

        if self.enable_delta_reward:
            self.last_reward = self.reward(self.current_state, self.output_text)

        return self.current_state, {}

    def _get_obs(self) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Get current observation, reward and done flags"""
        is_terminated = self.is_terminated(self.current_state)
        is_truncated = self.is_truncated(self.current_state)
        reward = self.reward(self.current_state, self.output_text)
        
        return self.current_state, reward, is_terminated, is_truncated, {}

    def render(self) -> str:
        """Render current state as text"""
        return self.tokenizer.decode(self.current_state['input_ids'])

    def close(self):
        pass

    def set_portion(self, portion):
        self.ground_truth_portion_dist = portion

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