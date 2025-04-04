import re
import numpy as np
import torch
from typing import Union, List, Dict
from transformers import PreTrainedTokenizer
from functools import partial
import warnings
import gc


def remove_filler_tokens_from_hashed_array(combined_tensor: torch.FloatTensor, filler_token: int) -> Dict[str, torch.Tensor]:
    """ Remove filler tokens from the combined tensor
    
    :param combined_tensor: Combined tensor
    :type combined_tensor: torch.FloatTensor
    :param filler_token: Filler token
    :type filler_token: int
    :return: Dictionary containing hidden states and input ids
    :rtype: Dict[str, torch.Tensor]
    """
    tensor_dict = unhash_ids_and_hidden_states(combined_tensor)
    input_ids = tensor_dict["input_ids"]
    #find postion of filler tokens in input_ids
    filler_positions = (input_ids == filler_token)
    return [combined_tensor[i][~filler_position] for i, filler_position in enumerate(filler_positions)]

def pad_hidden_states(last_hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor, filler_token: int, padding_side: str) -> torch.FloatTensor:
    """ Pad hidden states to make them of length max_seq_len
    
    :param last_hidden_states: Hidden states
    :type last_hidden_states: torch.FloatTensor
    :param max_seq_len: Maximum sequence length
    :type max_seq_len: int
    :param filler_token: Filler token
    :type filler_token: int
    :param padding_side: Padding side
    :type padding_side: str
    :return: Padded hidden states
    :rtype: torch.FloatTensor
    """
    seq_len_per_batch = (attention_mask.bool()).sum(dim = -1)
    # for some reason, if the seqence length is 1 then the shape of the last_hidden_states is (bs, hidden_dim) instead of (bs, 1, hidden_dim)
    
    
    if isinstance(last_hidden_states, np.ndarray):
        if len(last_hidden_states.shape) == 2:
            #unsqueeze on dimension 1
            last_hidden_states = last_hidden_states[:, np.newaxis, :]
        padded_last_hidden_states = np.full(
            (last_hidden_states.shape[0], attention_mask.shape[1], last_hidden_states.shape[2]),
            filler_token,
            dtype = last_hidden_states.dtype
        )

    elif isinstance(last_hidden_states, torch.Tensor):
        if len(last_hidden_states.shape) == 2:
            last_hidden_states = last_hidden_states.unsqueeze(1)
        padded_last_hidden_states = torch.full(
            (last_hidden_states.shape[0], attention_mask.shape[1], last_hidden_states.shape[2]),
            filler_token,
            device = last_hidden_states.device,
            dtype = last_hidden_states.dtype
        )
    else:
        raise ValueError("Array must be either a numpy array or a torch tensor")
    for idx, seq_len in enumerate(seq_len_per_batch):
        if seq_len == 0:
            continue
        elif padding_side == "right":
            padded_last_hidden_states[idx, :seq_len] = last_hidden_states[idx, :seq_len]
        else:
            padded_last_hidden_states[idx, -seq_len:] = last_hidden_states[idx, :seq_len]
    return padded_last_hidden_states

def hash_ids_and_hidden_states(input_ids: Union[np.ndarray, torch.LongTensor], last_hidden_states: Union[np.ndarray, torch.FloatTensor]) -> Union[np.ndarray, torch.FloatTensor]:
    # last_hidden_states -> (bs, seq_len, hidden_dim)
    # input_ids -> (bs, seq_len, 1)
    if isinstance(input_ids, torch.Tensor):
        cat_method = partial(torch.cat, dim = -1)
        input_ids = input_ids.unsqueeze(-1)
        tensors = (last_hidden_states, input_ids)
    elif isinstance(input_ids, np.ndarray):
        cat_method = partial(np.concatenate, axis = -1)
        input_ids = input_ids[..., np.newaxis]
        tensors = [last_hidden_states, input_ids]
    else:
        raise ValueError("Array must be either a numpy array or a torch tensor")
    
    combined_tensor = cat_method(tensors)
    return combined_tensor

def unhash_ids_and_hidden_states(combined_tensor: Union[np.ndarray, torch.FloatTensor]) -> Dict[str, Union[np.ndarray, torch.FloatTensor]]:
    if isinstance(combined_tensor, torch.Tensor):
        return {"last_hidden_states": combined_tensor[..., :-1], "input_ids": combined_tensor[..., -1].long()}
    elif isinstance(combined_tensor, np.ndarray):
        return {"last_hidden_states": combined_tensor[..., :-1], "input_ids": combined_tensor[..., -1].astype(int)}
    else:
        raise ValueError("Array must be either a numpy array or a torch tensor")

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


def add_filler_tokens(array: Union[np.ndarray, torch.Tensor], max_tokens: int, filler_token: int, dim = -1)-> Union[np.ndarray, torch.Tensor]:
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
        cat_method = partial(torch.cat, dim = dim)
        create_tensor_method = partial(torch.full, device = array.device)
    elif isinstance(array, np.ndarray):
        cat_method = partial(np.concatenate, axis = dim)
        create_tensor_method = np.full
    else:
        raise ValueError("Array must be either a numpy array or a torch tensor")
    
    if array.shape[dim] > max_tokens:
        warnings.warn(
            f"Array is already longer than max_tokens (max_tokens: {max_tokens}, your array length: {array.shape[-1]}). \
                Array will be truncated. If this is not the desired behavior, consider increasing the max_tokens parameter")
        array = array[..., :max_tokens]
    
    elif array.shape[dim] < max_tokens:
        array_shape = list(array.shape)[:dim]
        next_dim = max_tokens - array.shape[dim]
        array_shape.append(next_dim)
        if dim < len(array.shape) - 1 and dim != -1 :
            array_shape.extend(list(array.shape)[dim + 1:])
        
        filler_tensor = create_tensor_method(array_shape, filler_token)
        
        array = cat_method([array, filler_tensor])

    return array

def sync_and_clear_cuda_cache(device):
    torch.cuda.synchronize(device)  
    gc.collect()
    torch.cuda.empty_cache()
        

