
import sys
from typing import List,Dict,Union

sys.path.append("..",)
sys.path.append(".")
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
    def __init_(self, tokenizer):
        self.tokenizer = tokenizer
        
    def reward_fn(self, model_output: str, ground_truth: str):
        #an adaptation of thirdparty.openai.grade_school_math.dataset.is_correct
        gt_answer = extract_answer(strip_special_tokens(ground_truth,self.tokenizer))
        assert gt_answer != INVALID_ANS
        pred_answer = extract_answer(strip_special_tokens(model_output,self.tokenizer))
        if pred_answer == INVALID_ANS:
            return -1
        return int(pred_answer == gt_answer)
    
    def get_max_reward(self):
        return 1

class LogLikelihoodReward(AbstractReward):
    def __init__(self, tokenizer, model,tokens_ids_to_ignore,invalid_answer_penalty = torch.finfo(torch.float).min, use_conditional_logits=True):
        self.tokenizer = tokenizer
        self.model = model
        self.tokens_ids_to_ignore = tokens_ids_to_ignore
        self.invalid_ans_penalty = invalid_answer_penalty
        self.use_conditional_logits = use_conditional_logits

    def set_model(self, model):
        self.model = model
        
    def get_masked_output_logits(self, model_output: str, padding: bool):
        
        tokenized_seqs = self.tokenizer(model_output, return_tensors="pt", padding=padding, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_seqs, return_dict=True)
        
        logits_log_softmax = torch.nn.functional.log_softmax(outputs.logits[:,:-1,:], dim=-1)
        if not self.use_conditional_logits:
            logits_not_pause_log_softmax = torch.nn.functional.log_softmax(outputs['pause_logits'][:, :-1, :], dim=-1)[..., 0]
            logits_log_softmax += logits_not_pause_log_softmax.unsqueeze(-1)
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
        self.model.eval()
        masked_output_logits = self.get_masked_output_logits(model_output, padding=False)
        ll = masked_output_logits.sum()
        if was_in_training:
            self.model.train()
        return ll
 
    
    def batch_call(self, model_output: List[str], ground_truth: List[str]):
        
        answers = [strip_special_tokens(output,self.tokenizer) for output in model_output]
        invalid_ans_mask = [answer == INVALID_ANS for answer in answers]
        was_in_training = self.model.training
        self.model.eval()
        masked_output_logits = self.get_masked_output_logits(model_output,padding= True)
        ll = masked_output_logits.sum(dim=-1)
        ll[invalid_ans_mask] = self.invalid_ans_penalty
        if was_in_training:
            self.model.train()
        return ll
    
    def get_max_reward(self):
        #TODO: Probably change this, but not using it for now
        return -10
    

class GSM8KFinalAnswerLogLikelihoodReward(LogLikelihoodReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.tokens_ids_to_ignore.append(self.tokenizer.eos_token_id)
        self.tokens_ids_to_ignore.append(tokenizer.encode(' ')[0])

    def get_start_of_answer_token_position(self, model_output: str):
        #find the position of the first token of the answer
        identifier = self.tokenizer.encode('####')[0]
        tokenized_output = self.tokenizer(model_output, return_tensors="pt", padding=False, truncation=True)
        start_of_answer_token_position = torch.where(tokenized_output['input_ids'] == identifier)[1].item()
        return start_of_answer_token_position

    def get_masked_output_logits(self, model_output: str, padding: bool):
        tokenized_seqs = self.tokenizer(model_output, return_tensors="pt", padding=padding, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_seqs, return_dict=True)

        # discard the last token, it's model's output to EOS
        # logits_log_softmax = torch.nn.functional.log_softmax(outputs.logits[:,:-1,:], dim=-1)
        logits_log_softmax = torch.nn.functional.log_softmax(outputs['lm_logits'][:, :-1, :], dim=-1)
        if not self.use_conditional_logits:
            logits_not_pause_log_softmax = torch.nn.functional.log_softmax(outputs['pause_logits'][:, :-1, :], dim=-1)[..., 0]
            logits_log_softmax += logits_not_pause_log_softmax.unsqueeze(-1)
        # #get mask on tokens to ignore
        full_mask = torch.zeros_like(outputs.logits[:,:-1,:], dtype=torch.bool).to(self.model.device)
        full_mask[:,:self.get_start_of_answer_token_position(model_output)] = True
        for token_id in self.tokens_ids_to_ignore:
            token_mask = (tokenized_seqs['input_ids'][:,1:] == token_id)
            full_mask[token_mask,:] = True
        
        logits_log_softmax[full_mask] = 0.0
        #fetch output of logits_log_softmax using tokenized_seqs['input_ids']
        model_output_logits = torch.gather(logits_log_softmax, dim=-1, index=tokenized_seqs['input_ids'][:,1:].unsqueeze(-1)).squeeze(-1)
    
        return model_output_logits 
    
    

class LogLikelihoodRewardWithPausePenalty(LogLikelihoodReward):
    def __init__(self, pause_str = "<|pause|>" ,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pause_str = pause_str
    
    def n_tokens_penalty(self,model_output):
        
        if isinstance(model_output, str):
            #count how many times pause_str appears in model_output
            n_pauses = model_output.count(self.pause_str)
            penalty = 6.0 / (torch.pi ** 2) * (1.0 / (n_pauses + 1)**2)
            
        
        elif isinstance(model_output, list):
            n_pauses = [output.count(self.pause_str) for output in model_output]
            penalty = [6.0 / (torch.pi ** 2 )* (1.0 / (n_pause + 1)**2) for n_pause in n_pauses]    
        return torch.log(torch.tensor(penalty))
    
    def reward_fn(self, model_output: str, ground_truth: str):
        ll = super().reward_fn(model_output,ground_truth)
        penalty = self.n_tokens_penalty(model_output)
        return ll + penalty
            
    def batch_call(self, model_output: List[str], ground_truth: List[str]):
        ll = super().batch_call(model_output,ground_truth)
        penalty = self.n_tokens_penalty(model_output)
        
        return ll + penalty

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
    
    
    
if __name__ == '__main__':
    import transformers
    from tokenizers import AddedToken
    from pause_classifier_wrapper import PauseClassifierWrapper, PauseCLFConfig
    
    lm =  transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    
    pause_token = AddedToken(
        "<|pause|>", 
        single_word=False, 
        # lstrip=True, 
        # rstrip=True
    )
    
    tokenizer.add_tokens([pause_token], special_tokens=True)
    lm.resize_token_embeddings(len(tokenizer))
    pause_token_id = tokenizer.convert_tokens_to_ids("<|pause|>")
    pause_clf_config = PauseCLFConfig(
        pause_token_id=pause_token_id,
    )
    model = PauseClassifierWrapper(pause_clf_config,lm)
    
    # reward = LogLikelihoodReward(
    #         tokenizer=tokenizer,
    #         model=model,
    #         tokens_ids_to_ignore=[pause_token_id],
    #     )
    
    reward = GSM8KFinalAnswerLogLikelihoodReward(
            tokenizer=tokenizer,
            model=model,
            tokens_ids_to_ignore=[pause_token_id],
            use_conditional_logits=False
    )
    
    mock_sentence =  " He saved up $110 total because 95 + 15 =<|pause|> <<95<|pause|>+15=110>>1<|pause|>1<|pause|>0\nHe saved $15 from his allowance because 3 x 5 = <<3<|pause|>*5=15>>15\nHe earned $60 mowing lawns because 4 x 15 = <<4*15=60>>6<|pause|>0\nHe earned $35 shoveling driveways because 110 - 60 - 15 = <<110-60-15=35>>35\nHe shoveled 5 driveways because 3<|pause|>5 / 7 = <<3<|pause|>5/7=5>>5\n####<|pause|> 5 <|endoftext|>"
    gt = None    
    print(reward(mock_sentence, gt))
    
