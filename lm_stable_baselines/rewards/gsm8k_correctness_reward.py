from lm_stable_baselines.rewards import AbstractReward
import torch
from lm_stable_baselines.utils import strip_special_tokens, extract_answer, INVALID_ANS 

class GSM8KCorrectnessReward(AbstractReward):
    """ Reward function for the GSM8K dataset. This reward function checks if the model output is correct given the ground truth answer. The ground truth answer must be in the GSM8K dataset format
    
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def reward_fn(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor):
        """An adaptation of thirdparty.openai.grade_school_math.dataset.is_correct. Reward function, returns 1.0 if the model output is correct w.r.t. ground truth, 0.0 if it is incorrect, -1.0 if the model output is invalid
        
        :param model_output: Model output
        :type model_output: torch.LongTensor
        :param ground_truth: Ground truth
        :type ground_truth: torch.LongTensor
        :return: Reward
        :rtype: float
        """
        #an adaptation of thirdparty.openai.grade_school_math.dataset.is_correct
        
        #extract the answer of gt
        gt_answer = extract_answer(strip_special_tokens(ground_truth,self.tokenizer))
        assert gt_answer != INVALID_ANS, f"Ground truth answer is invalid, your ground truth answer is {ground_truth}"
        #extract the answer of the model output
        pred_answer = extract_answer(strip_special_tokens(model_output,self.tokenizer))
        
        #if the model output is invalid, return -1.0
        if pred_answer == INVALID_ANS:
            return -1.0
        #if the model output is correct, return 1.0 otherwise return 0.0
        return float(pred_answer == gt_answer)
    
    def get_max_reward(self):
        """ Get the maximum reward value (1.0)
        
        :return: Maximum reward value
        :rtype: float
        """
        return 1.0
    
    def get_min_reward(self):
        """ Get the minimum reward value (-1.0)
        
        :return: Minimum reward value
        :rtype: float
        """
        return -1.0