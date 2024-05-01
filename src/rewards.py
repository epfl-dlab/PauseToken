
import sys
from typing import List,Dict,Union
sys.path.append("..")
from thirdparty.openai.grade_school_math.dataset import extract_answer,INVALID_ANS
import math
class AbstractReward:
    def __call__(self, model_output: Union[str,List[str]], ground_truth: Union[str,List[str]]):
        if isinstance(model_output, str):
            return self.reward_fn(model_output, ground_truth)
        elif isinstance(model_output, list):
            return [
                self.reward_fn(model_output, ground_truth) 
                for model_output, ground_truth in zip(model_output, ground_truth)
            ]
        else:
            raise ValueError("model_output and ground_truth must be either a string or a list of strings")    
    def reward_fn(self,model_output: str, ground_truth: str):
        raise NotImplementedError
        
    def get_max_reward(self):
        raise NotImplementedError

class GSM8KCorrectnessReward(AbstractReward):
    
    def reward_fn(self, model_output: str, ground_truth: str):
        #an adaptation of thirdparty.openai.grade_school_math.dataset.is_correct
        gt_answer = extract_answer(ground_truth)
        assert gt_answer != INVALID_ANS
        pred_answer = extract_answer(model_output)
        if pred_answer == INVALID_ANS:
            return -1
        return int(pred_answer == gt_answer)
    
    def get_max_reward(self):
        return 1
    

class GSM8KDeltaAnswerReward(AbstractReward):
    
    def __init__(self, invalid_ans_penalty: float = -1000, ceiled = True):
        self.invalid_ans_penalty = invalid_ans_penalty
        self.ceiled = ceiled
    
    def reward_fn(self, model_output: str, ground_truth: str):
        model_output = extract_answer(model_output)
        
        if model_output == INVALID_ANS:
            return self.invalid_ans_penalty
        
        ground_truth = extract_answer(ground_truth)
        delta = -abs(float(model_output) - float(ground_truth))
        if self.ceiled:
            delta = -int(math.ceil(abs(delta)))
        return delta
    
    def get_max_reward(self):
        return 0