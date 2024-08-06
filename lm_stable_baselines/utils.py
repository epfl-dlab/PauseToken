import re
import numpy as np
import torch
from typing import Union, List
from transformers import PreTrainedTokenizer
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def remove_filler_tokens(obs: torch.Tensor, filler_token: int) -> Union[torch.Tensor, List[torch.Tensor]]:
    """ Remove filler tokens from the obs tensor. Function usually used before padding
    
    :param obs: Observation tensor
    :type obs: torch.Tensor
    :param filler_token: Filler token
    :type filler_token: int
    :return: Observation tensor without filler tokens, returns either a 2D tensor or a list of 1D tensors
    :rtype: Union[torch.Tensor, List[torch.Tensor]]
    """
    #check for any filler tokens
    if not (obs == filler_token).any():
        return obs
    
    shape = obs.shape
    #if it is a 1D tensor we can filter it directly
    if len(shape) == 1:
        return obs[obs != filler_token].reshape(-1,1)
    #If it is a 2D tensor we have to filter each row and return a list of 1D tensors
    return [ob[ob != filler_token] for ob in obs]


def add_filler_tokens(array: np.array, max_tokens: int, filler_token: int)-> np.array:
    """ Add filler tokens to the array to make it of length max_tokens
    
    :param array: Array to add filler tokens to
    :type array: np.array
    :param max_tokens: Maximum number of tokens
    :type max_tokens: int
    :param filler_token: Filler token
    :type filler_token: int
    :return: Array with filler tokens
    :rtype: np.array
    """
    if array.shape[-1] < max_tokens:
        return np.concatenate([array, np.array([filler_token] * (max_tokens - array.shape[-1]))])
    return array

def extract_answer(completion: str) -> str:
    """ Extracts the answer from the completion following the GSM8K dataset format
    
    :param completion: Completion
    :type completion: str
    :return: Extracted answer
    :rtype  str
    """
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion: str, gt_example: str) -> bool:
    """ Check if the model completion is correct given the ground truth example. Completions must be in the GSM8K dataset format
    
    :param model_completion: Model completion
    :type model_completion: str
    
    """
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS, \
        f"Ground truth answer is invalid and doesn't follow the GSM8K formate, your ground truth answer is {gt_example['answer']}"
    return extract_answer(model_completion) == gt_answer

def strip_special_tokens(input_ids: Union[int, List[int], np.ndarray, torch.Tensor], tokenizer: PreTrainedTokenizer) -> str:
    """ Strip special tokens from a text
    
    :param input_ids: Input ids
    :type text: Union[int, List[int], np.ndarray, torch.Tensor]
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :return: Text without special tokens
    :rtype: str
    """
    return tokenizer.decode(input_ids, skip_special_tokens=True)