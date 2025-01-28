from src.reward import GSM8KCorrectnessReward
from src.utils.trainer_utils import decode_and_strip_special_tokens
import torch

def extract_answer(text: str) -> str:
    """Extract the answer from the text
    
    :param text: Text
    :type text: str
    :return: Answer
    :rtype: str
    """
    #an adaptation of thirdparty.openai.grade_school_math.dataset.extract_answer
    #extract the answer from the text
    answer = text.split("####")[-1]
    return answer.strip()
    #if the answer is not a digit
    
def is_correct(model_completion: str, gt_example: str) -> bool:
    """ Check if the model completion is correct given the ground truth example. Completions must be in the GSM8K dataset format
    
    :param model_completion: Model completion
    :type model_completion: str
    
    """
    gt_answer = extract_answer(gt_example)
    pred_answer = extract_answer(model_completion)
    return (pred_answer in gt_answer)

class ProQACorrectnessReward(GSM8KCorrectnessReward):
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
        gt_answer = extract_answer(decode_and_strip_special_tokens(ground_truth,self.tokenizer))
        #extract the answer of the model output
        pred_answer = extract_answer(decode_and_strip_special_tokens(model_output,self.tokenizer))            
        #if the model output is correct, return 1.0 otherwise return 0.0
        return float(pred_answer == gt_answer)