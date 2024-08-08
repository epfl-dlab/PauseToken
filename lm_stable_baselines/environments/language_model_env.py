from typing import Tuple, Any
from gymnasium import Env,spaces
from typing import List, Dict
from transformers import PreTrainedTokenizer
from datasets import Dataset
from lm_stable_baselines.rewards import AbstractReward
import numpy as np

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
    dataset = None 
    def __init__(
        self,
        reward: AbstractReward,
        tokenizer: PreTrainedTokenizer,
        termination_tokens: List[int],
        max_tokens: int,
        dataset: Dataset = None,
        require_dataset: bool = False,
        filler_token: int = -100,
    ):
        super(LanguageModelEnv, self).__init__()

        self.reward = reward
        self.termination_tokens = termination_tokens
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.filler_token = filler_token
        
        if require_dataset and not LanguageModelEnv.dataset:
            if dataset is None:
                raise ValueError("dataset must be provided")
            LanguageModelEnv.dataset = dataset
        
        self.observation_space =  spaces.MultiDiscrete([tokenizer.vocab_size]* max_tokens, dtype = np.int64) 
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        self.current_state = []

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """ Apply an action to the environment. For a language model it's simply adding the action to the current state
        
        :param action: Action to apply
        :type action: int
        :return: Observation, reward, termination signal, truncation signal, info
        :rtype: Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
        """
        self.current_state.append(action.item())
        observation , reward, terminated, truncated, info = self._get_obs()
        return observation, reward, terminated, truncated, info

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
        if not self.is_terminated(state) and len(state) >= self.max_tokens:
            return True
        return False
    
    def reset(
        self,
        seed = 123,
        id: int = None,
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
        if id is None:
            id = self.np_random.choice(len(LanguageModelEnv.dataset))
        input_sample = LanguageModelEnv.dataset[id]
        input_text = input_sample["input_text"]
        #save the output text (ground truth)
        self.output_text = self.tokenizer(input_sample["output_text"], return_tensors="np", padding=True, truncation=True)["input_ids"].reshape(-1).tolist()
        batch_encoding = self.tokenizer(input_text, return_tensors="np", padding=True, truncation=True)
        #save the current state (input text)
        self.current_state = batch_encoding["input_ids"].reshape(-1).tolist()
        
        #return the observation and info
        self.last_obs = self.current_state
        self.terminated = False
        self.truncated = False
        self.done = False
        return np.array(self.current_state), {} # return observation and info=None
    
    def _get_obs(self):
        """ Get the observation, reward, termination signal, truncation signal and info
        
        :return: Observation, reward, termination signal, truncation signal, info
        :rtype: Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
        """
        is_terminated =  self.is_terminated(self.current_state)
        
        is_truncated = self.is_truncated(self.current_state)
        reward = self.reward(self.current_state, self.output_text) if is_terminated or is_truncated else self.reward.get_min_reward()
        
        info = {}
        self.last_obs = self.current_state
        full_current_state = self.current_state #self.resize_obs()
       
        return np.array(full_current_state) , reward, is_terminated, is_truncated, info

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