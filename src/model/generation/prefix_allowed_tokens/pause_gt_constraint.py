from typing import Optional, Callable, Iterable
import torch
import random

class PauseGroundTruthConstraint:
    """ Constraint for sampling either the pause token or the next token in the ground truth. It also ensures that the whole ground truth is generated within the max_tokens limit.
    
    :param tokens_to_filter: Tokens to filter out from the prefix
    :type tokens_to_filter: List[int]
    :param max_tokens: Maximum number of tokens that can be generated
    :type max_tokens: int
    """
    def __init__(self, tokens_to_filter,max_tokens,allowed_tokens_to_predict):
        self.tokens_to_filter = tokens_to_filter
        self.allowed_tokens_to_predict = allowed_tokens_to_predict
        self.max_tokens = max_tokens
    
    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        
        return _get_prefix_allowed_tokens_fn(
                tokenized_ground_truths=batch_info["batch_info"].get("tokenized_ground_truths",[""]),
                allowed_tokens_to_predict=self.allowed_tokens_to_predict,
                pad_token_id=batch_info["batch_info"]["pad_token_id"],
                tokens_to_filter = self.tokens_to_filter,
                max_tokens=self.max_tokens,
                pause_prob=batch_info["batch_info"].get("pause_prob",None)
            )

def _get_prefix_allowed_tokens_fn(tokenized_ground_truths, allowed_tokens_to_predict, pad_token_id,tokens_to_filter,max_tokens, pause_prob=None):
    ##NOTE: pause_prob is yet not utilized
    
    
    tokenized_gts_dict = {i: tokenized_gt[tokenized_gt != pad_token_id] for i, tokenized_gt in enumerate(tokenized_ground_truths)}   
    
    def compute_pause_budget(tokenized_gt, prefix, filtered_prefix):
        """ More or less computes the remaining pause budget given the prefix. Most importantly, it returns the model can pause or not."""
        if max_tokens is None:
            return 1
        remaining_gts = len(tokenized_gt) - len(filtered_prefix)
        if remaining_gts <= 0:
            return 0
        remaining_pause_budget = max_tokens - len(prefix) - remaining_gts
        return remaining_pause_budget
        
    def get_allowed_tokens(prefix,batch_id):
       
        #filter_out_pause_token_id_from_prefix
        filtered_prefix = \
            list(
                filter(
                    lambda token_id: token_id not in tokens_to_filter,
                    prefix
                )
            )
        #get corresponding tokenized ground truth
        tokenized_gt = tokenized_gts_dict[batch_id]
        #check if pause budget exceeded
        pause_budged_exceeded = compute_pause_budget(tokenized_gt, prefix, filtered_prefix) <= 0
        
        #Note: pause_prob is not yet utilized
        force_random_p = False if pause_prob is None else random.random() < pause_prob
        
        #if the whole ground truth is generated
        if len(filtered_prefix) >= len(tokenized_gt):
            next_possible_token = [pad_token_id]
        
        #If we're still allowed to pause
        elif not pause_budged_exceeded:
            ## Ignor this for the moment. It's not yet utilized
            if pause_prob is not None:
                if force_random_p:
                    next_possible_token = allowed_tokens_to_predict
                else:
                    next_possible_token = [tokenized_gt[len(filtered_prefix)]]
                    
            #If we're still allowed to pause, the model can either predict the next ground truth token or pause
            else:
                next_possible_token = [tokenized_gt[len(filtered_prefix)]]
                next_possible_token.extend(allowed_tokens_to_predict) 
        #If we're not allowed to pause (budget exceeded), the model can only predict the next ground truth token
        else:
            next_possible_token = [tokenized_gt[len(filtered_prefix)]] 
        
        return next_possible_token

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> Iterable[int]:
        #convert tensor to list
        prefixes = input_ids.tolist()
        return get_allowed_tokens(prefixes,batch_id)

    return prefix_allowed_tokens_fn
