from typing import Optional, Callable, Iterable
import torch
import random
from transformers import LogitsProcessor


class PauseLogitsProcessor(LogitsProcessor):
    """ LogitsProcessor for Pause Tokens. It processes the logits to increase (pause_temperature > 1) or decrease (pause_temperature < 1) the probability of the pause token being sampled. 
    If pause_temperature = 0, Then it's never sampled

    :param pause_token_id: Pause token id
    :type pause_token_id: int
    :param pause_temperature: Pause temperature
    :type pause_temperature: float
    :param softmax_temperature: Softmax temperature (temperature used to sample from the model's logits) (should be the same as the HF generation params)
    :type softmax_temperature: float
    :return: Processed logits
    :rtype: torch.FloatTensor
    """
    
    def __init__(self, pause_token_id, pause_temperature, softmax_temperature):
        self.pause_token_id = pause_token_id
        self.pause_temperature = pause_temperature
        self.softmax_temperature = softmax_temperature
        
    # TODO: batching
    def process_logits(self, input_ids, scores):
        probs = torch.nn.functional.softmax(1/self.softmax_temperature * scores, dim=-1)
        pause_prob = probs[..., self.pause_token_id].clone().unsqueeze(-1)
        pause_odds = pause_prob / (1 - pause_prob)
        pause_new_prob = self.pause_temperature * pause_odds/(1 + self.pause_temperature * pause_odds)
        
        mask = torch.ones_like(probs, dtype=torch.bool)
        is_nan_condition = torch.isnan(pause_new_prob).squeeze(-1)
        full_prob_on_pause = torch.where(is_nan_condition)[0]
        mask[full_prob_on_pause,:] = False
        probs = torch.where(mask ,probs * (1- pause_new_prob)/ (1 - pause_prob), probs)
        probs[:, self.pause_token_id] = torch.where(is_nan_condition, 1.0 , pause_new_prob.squeeze(-1))
        log_probs = self.softmax_temperature * torch.log(probs)
        #check if `inf`, `nan` are in probs
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            raise ValueError("LogitsProcessor: the model generated `inf` or `nan` values")
        return log_probs
                
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)

class PauseGroundTruthConstraint:
    """ Constraint for sampling either the pause token or the next token in the ground truth. It also ensures that the whole ground truth is generated within the max_tokens limit.
    
    :param tokens_to_filter: Tokens to filter out from the prefix
    :type tokens_to_filter: List[int]
    :param max_tokens: Maximum number of tokens that can be generated
    :type max_tokens: int
    """
    def __init__(self, tokens_to_filter,max_tokens):
        self.tokens_to_filter = tokens_to_filter
        self.max_tokens = max_tokens
    
    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        
        return _get_prefix_allowed_tokens_fn(
                tokenized_ground_truths=batch_info["batch_info"].get("tokenized_ground_truths",[""]),
                pause_token_id=batch_info["batch_info"]["pause_token_id"],
                pad_token_id=batch_info["batch_info"]["pad_token_id"],
                tokens_to_filter = self.tokens_to_filter,
                max_tokens=self.max_tokens,
                pause_prob=batch_info["batch_info"].get("pause_prob",None)
            )

def _get_prefix_allowed_tokens_fn(tokenized_ground_truths, pause_token_id, pad_token_id,tokens_to_filter,max_tokens, pause_prob=None):
    ##NOTE: pause_prob is yet not utilized
    
    
    tokenized_gts_dict = {i: tokenized_gt for i, tokenized_gt in enumerate(tokenized_ground_truths)}   
    
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
                    next_possible_token = [pause_token_id]
                else:
                    next_possible_token = [tokenized_gt[len(filtered_prefix)]]
                    
            #If we're still allowed to pause, the model can either predict the next ground truth token or pause
            else:
                next_possible_token = [tokenized_gt[len(filtered_prefix)], pause_token_id] 
        #If we're not allowed to pause (budget exceeded), the model can only predict the next ground truth token
        else:
            next_possible_token = [tokenized_gt[len(filtered_prefix)]] 
        
        return next_possible_token

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> Iterable[int]:
        #convert tensor to list
        prefixes = input_ids.tolist()
        return get_allowed_tokens(prefixes,batch_id)

    return prefix_allowed_tokens_fn
