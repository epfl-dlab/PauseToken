
import sys
from typing import List,Dict,Union
sys.path.append("..")
from thirdparty.openai.grade_school_math.dataset import extract_answer,INVALID_ANS
import math
import torch
from utils import strip_special_tokens
class AbstractReward:
    def __call__(self, model_output: Union[str,List[str]], ground_truth: Union[str,List[str]]):
        if isinstance(model_output, str):
            return self.reward_fn(model_output, ground_truth)
        elif isinstance(model_output, list):
            return self.batch_call(model_output, ground_truth)
        else:
            raise ValueError("model_output and ground_truth must be either a string or a list of strings")
    
    def batch_call(self, model_output: List[str], ground_truth: List[str]):    
        return [
                self.reward_fn(model_output, ground_truth) 
                for model_output, ground_truth in zip(model_output, ground_truth)
            ]
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
    

class LogLikelihoodReward(AbstractReward):
    def __init__(self, tokenizer, model,tokens_ids_to_ignore,invalid_answer_penalty = torch.finfo(torch.float).min):
        self.tokenizer = tokenizer
        self.model = model
        self.tokens_ids_to_ignore = tokens_ids_to_ignore
        self.invalid_ans_penalty = invalid_answer_penalty
        
    def set_model(self, model):
        self.model = model
        
    def get_masked_output_logits(self, model_output: str, padding: bool):
        
        tokenized_seqs = self.tokenizer(model_output, return_tensors="pt", padding=padding, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_seqs, return_dict=True)
        
        logits_log_softmax = torch.nn.functional.log_softmax(outputs.logits[:,:-1,:], dim=-1)
        # #get mask on tokens to ignore
        full_mask = torch.zeros_like(outputs.logits[:,:-1,:], dtype=torch.bool).to(self.model.device)
        for token_id in self.tokens_ids_to_ignore:
            token_mask = (tokenized_seqs['input_ids'][:,1:] == token_id)
            full_mask[token_mask,:] = True
        
        logits_log_softmax[full_mask] = 0.0
        #fetch output of logits_log_softmax using tokenized_seqs['input_ids']
        model_output_logits = torch.gather(logits_log_softmax, dim=-1, index=tokenized_seqs['input_ids'][:,1:].unsqueeze(-1)).squeeze(-1)
    
        return model_output_logits 
    
    def reward_fn(self, model_output: str, ground_truth: str):
        
        answer = extract_answer(strip_special_tokens(model_output,self.tokenizer))

        if answer == INVALID_ANS:
            return self.invalid_ans_penalty
        
        was_in_training = self.model.training
        masked_output_logits = self.get_masked_output_logits(model_output, padding=False)
        ll = masked_output_logits.sum()
        if was_in_training:
            self.model.train()
        return ll
 
    
    def batch_call(self, model_output: List[str], ground_truth: List[str]):
        
        answers = [strip_special_tokens(output,self.tokenizer) for output in model_output]
        invalid_ans_mask = [answer == INVALID_ANS for answer in answers]
        was_in_training = self.model.training
        masked_output_logits = self.get_masked_output_logits(model_output,padding= True)
        ll = masked_output_logits.sum(dim=-1)
        ll[invalid_ans_mask] = self.invalid_ans_penalty
        if was_in_training:
            self.model.train()
        return ll
    
    def get_max_reward(self):
        #TODO: Probably change this, but not using it for now
        return -10
    
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