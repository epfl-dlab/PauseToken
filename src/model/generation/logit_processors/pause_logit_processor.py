from transformers import LogitsProcessor
import torch

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
    
    def __init__(self, pause_token_id, constraint_module):
        self.pause_token_id = pause_token_id
        self.gt_constraint = constraint_module 
        
    def set_prefix_allowed_tokens_fn(self, batch_info):
        self.prefix_allowed_tokens_fn = self.gt_constraint.get_prefix_allowed_tokens_fn(batch_info=batch_info)
        # self.pause_temperature = pause_temperature
        # self.softmax_temperature = softmax_temperature
        
    # TODO: batching
    def process_logits(self, input_ids, scores):
        allowed_tokens = [self.prefix_allowed_tokens_fn(batch_id = i, input_ids = input_ids[i]) for i in range(input_ids.size(0))]
        can_pause = [self.pause_token_id in tokens for tokens in allowed_tokens]
        
        probs = torch.nn.functional.softmax(scores, dim=-1)
        pause_prob = probs[..., self.pause_token_id].clone().unsqueeze(-1)
        
        #decide whether to pause or not in function of pause prob
        pause = torch.bernoulli(pause_prob).bool().squeeze(-1).squeeze(-1) & torch.tensor(can_pause, dtype=torch.bool).to(pause_prob.device)
        new_scores = scores.clone()
        
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for sample_id, tokens in enumerate(allowed_tokens):
            mask[sample_id, tokens] = True
        
        new_scores = torch.where(mask, new_scores, float("-inf"))
        new_scores[:, self.pause_token_id] = torch.where(pause, 0.0, float("-inf"))
        new_scores[pause, :self.pause_token_id] = float("-inf")
        new_scores[pause, self.pause_token_id+1:] = float("-inf")
        return new_scores
        
        # mask = torch.ones_like(probs, dtype=torch.bool)
        # is_nan_condition = torch.isnan(pause_new_prob).squeeze(-1)
        # full_prob_on_pause = torch.where(is_nan_condition)[0]
        # mask[full_prob_on_pause,:] = False
        # probs = torch.where(mask ,probs * (1- pause_new_prob)/ (1 - pause_prob), probs)
        # probs[:, self.pause_token_id] = torch.where(is_nan_condition, 1.0 , pause_new_prob.squeeze(-1))
        # log_probs = self.softmax_temperature * torch.log(probs)
        # #check if `inf`, `nan` are in probs
        # if torch.isnan(probs).any() or torch.isinf(probs).any():
        #     raise ValueError("LogitsProcessor: the model generated `inf` or `nan` values")
        # return log_probs
                
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)