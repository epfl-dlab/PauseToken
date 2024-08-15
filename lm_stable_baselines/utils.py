import re
import numpy as np
import torch
from typing import Union, List
from transformers import PreTrainedTokenizer
from functools import partial
import warnings


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
        return obs.reshape(-1,1) if len(obs.shape) == 1 else [ob for ob in obs]
    
    shape = obs.shape
    #if it is a 1D tensor we can filter it directly
    if len(shape) == 1:
        return obs[obs != filler_token].reshape(-1,1)
    #If it is a 2D tensor we have to filter each row and return a list of 1D tensors
    return [ob[ob != filler_token] for ob in obs]


def add_filler_tokens(array: Union[np.ndarray, torch.Tensor], max_tokens: int, filler_token: int)-> Union[np.ndarray, torch.Tensor]:
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
    if isinstance(array, torch.Tensor):
        cat_method = partial(torch.cat, dim = -1)
        create_tensor_method = partial(torch.full, device = array.device)
    elif isinstance(array, np.ndarray):
        cat_method = partial(np.concatenate, axis = -1)
        create_tensor_method = np.full
    else:
        raise ValueError("Array must be either a numpy array or a torch tensor")
    
    if array.shape[-1] >= max_tokens:
        warnings.warn(
            f"Array is already longer than max_tokens (max_tokens: {max_tokens}, your array length: {array.shape[-1]}). \
                Array will be truncated. If this is not the desired behavior, consider increasing the max_tokens parameter")
        array = array[..., :max_tokens]
        
    if array.shape[-1] < max_tokens:
        array_shape = list(array.shape)[:-1]
        last_dim = max_tokens - array.shape[-1]
        filler_tensor = create_tensor_method(tuple(array_shape + [last_dim]), filler_token)
        
        array = cat_method([array, filler_tensor])

    return array
        

