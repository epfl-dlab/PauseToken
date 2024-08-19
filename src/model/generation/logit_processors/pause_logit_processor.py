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