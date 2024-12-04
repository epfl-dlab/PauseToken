from lm_stable_baselines.rewards import AbstractReward
import torch
from transformers import PreTrainedTokenizer
from src.utils.trainer_utils import decode_and_strip_special_tokens, extract_answer, INVALID_ANS 
class GSM8KCorrectnessReward(AbstractReward):
    """Reward function for the GSM8K dataset. This reward function checks if the model output is correct given the ground truth answer. The ground truth answer must be in the GSM8K dataset format
    
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        
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
        assert gt_answer != INVALID_ANS, f"Ground truth answer is invalid, your ground truth answer is {ground_truth}"
        #extract the answer of the model output
        pred_answer = extract_answer(decode_and_strip_special_tokens(model_output,self.tokenizer))
        
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
    
class GSM8KFinalAnswerLogLikelihoodReward(AbstractReward):
    def __init__(
        self,
        tokenizer,
        # tokens_ids_to_ignore,
        model = None,
        delimiter= "####",
        # invalid_answer_penalty = torch.finfo(torch.float).min,
        correctness_reward_weight = 20.0,
        **kwargs
    ):
        """ Reward function for the GSM8K dataset. This reward function computes the log likelihood of the final answer given the model output.
        To compute the log likelihood, we remove the answer after the delimiter and compute the log likelihood of the correct answer given the model output.
        """
        super().__init__(tokenizer = tokenizer, **kwargs)
        
        self.tokenizer = tokenizer
        self.model = model
        self.correctness_reward_weight = float(correctness_reward_weight)
        if correctness_reward_weight != 0:
            self.correctness_reward = GSM8KCorrectnessReward(tokenizer)
        # self.tokens_ids_to_ignore = tokens_ids_to_ignore
        # self.invalid_ans_penalty = invalid_answer_penalty
        
        # self.tokens_ids_to_ignore.append(self.tokenizer.eos_token_id)
        # self.tokens_ids_to_ignore.append(self.tokenizer.encode(' ')[-1])
        self.identifiers = self.find_all_token_ids(delimiter)
        self.model_peft_name = None
            
    def set_model(self, model):
        self.model = model
        
    def set_model_peft_name(self, model_peft_name):
        self.model_peft_name = model_peft_name
        
    def get_start_of_answer_token_position(self, model_output: torch.LongTensor):
        #find the position of the first token of the answer
        start_of_answer_token_positions = torch.where(torch.isin(model_output, self.identifiers))
        #check if it's empty
        found_match = len(start_of_answer_token_positions[0]) > 0
        start_of_answer_token_position = None

        if found_match:
            #find first match (first one because that's how we compute accuracy in the GSM8K dataset)
            start_of_answer_token_position = start_of_answer_token_positions[0][0].item()
        return start_of_answer_token_position
    
    def find_all_token_ids(self,string: str):
        #iterate through tokenizer tokens
        matching_token_ids = []
        for token,id in self.tokenizer.get_vocab().items():
            if string in token[-len(string):]:
                matching_token_ids.append(id)
                
        return torch.tensor(matching_token_ids)
    
    def reward_fn(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor):
        
        if self.model_peft_name is not None:
            og_active_adapter = self.model.active_adapter
            
            if self.model_peft_name == "disable peft":
                self.model.disable_adapter_layers()
            else:
                self.model.set_adapter(self.model_peft_name)
        start_of_answer_token_position_model_output = \
            self.get_start_of_answer_token_position(model_output)

        start_of_answer_token_position_ground_truth = \
            self.get_start_of_answer_token_position(ground_truth)

        if start_of_answer_token_position_ground_truth is None:
            raise ValueError("Ground truth answer does not contain the delimiter")
        
        #get true answer
        start_of_answer_token_position_ground_truth = start_of_answer_token_position_ground_truth + 1
        
        flag = start_of_answer_token_position_model_output is not None
        #get model output excluding the answer
        if flag:
            start_of_answer_token_position_model_output = start_of_answer_token_position_model_output + 1 
        else:
            #keep delimiter in answer if it's not in the answer
            start_of_answer_token_position_ground_truth -= 1
        
        ground_truth_answer = ground_truth[start_of_answer_token_position_ground_truth:]
        context_model_output =model_output[:start_of_answer_token_position_model_output]
        #concatenate the context and the ground truth answer
        input_ids = torch.cat((context_model_output, ground_truth_answer), dim=-1)
        
        # I want to include the delimiter in the context rather than the answer
        context_size = context_model_output.shape[0]
        if flag:
            context_size += 1

        true_ans_log_prop = self.final_answer_log_likelihood(input_ids.unsqueeze(0), context_size)
        
        
        if self.model_peft_name is not None:
            if self.model_peft_name == "disable peft":
                self.model.enable_adapter_layers()
            
            self.model.set_adapter(og_active_adapter)
        
        if flag and self.correctness_reward_weight != 0:
            correctness_reward = self.correctness_reward.reward_fn(model_output, ground_truth)
            return self.correctness_reward_weight * correctness_reward + true_ans_log_prop[0].item()
        
        
        return true_ans_log_prop[0].item()
        
    def final_answer_log_likelihood(self, input_ids: torch.LongTensor, context_length: int):
        attention_mask = torch.ones_like(input_ids)
        attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(
                **{
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                },
                return_dict=True
            )
        logits_log_softmax = torch.nn.functional.log_softmax(outputs.logits[:, context_length-1:-1, :], dim=-1)
        final_answer_lprob = torch.gather(logits_log_softmax, dim=-1, index=input_ids[:, context_length:].unsqueeze(-1)).squeeze(-1)
        return final_answer_lprob.sum(dim=-1)

    
    
class InsertNPauses(AbstractReward):
    def __init__(self, tokenizer, n_pauses, pause_token_str = "<|pause|>", **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.n_pauses = n_pauses
        self.pause_token_str = pause_token_str
        
    def reward_fn(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor):
        decoded_output = self.tokenizer.decode(model_output)
        #count how many pauses are in decoded_output
        n_pauses = decoded_output.count(self.pause_token_str)
        
        reward = (n_pauses - self.n_pauses)
        
        if reward > 0:
            reward = -reward
            
        return reward
        
    
    def get_max_reward(self):
        return 0.0
    
    def get_min_reward(self):
        return -1000
    # def batch_call(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor) -> List[float]:
    #     raise NotImplementedError("batch_call not implemented for GSM8KFinalAnswerLogLikelihoodReward would need to fix get_start_of_answer_token_position")

    
    # def get_masked_output_logits(self, model_output: str, padding: bool):
    #     tokenized_seqs = self.tokenizer(model_output, return_tensors="pt", padding=padding, truncation=True).to(self.model.device)
        
    #     with torch.no_grad():
    #         outputs = self.model(**tokenized_seqs, return_dict=True)

    #     # discard the last token, it's model's output to EOS
    #     # logits_log_softmax = torch.nn.functional.log_softmsax(outputs.logits[:,:-1,:], dim=-1)
 
    #     logits_log_softmax = torch.nn.functional.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    #     # #get mask on tokens to ignore
    #     full_mask = torch.zeros_like(outputs.logits[:,:-1,:], dtype=torch.bool).to(self.model.device)
    #     full_mask[:,:self.get_start_of_answer_token_position(model_output)] = True
    #     for token_id in self.tokens_ids_to_ignore:
    #         token_mask = (tokenized_seqs['input_ids'][:,1:] == token_id)
    #         full_mask[token_mask,:] = True
        
    #     logits_log_softmax[full_mask] = 0.0
    #     #fetch output of logits_log_softmax using tokenized_seqs['input_ids']
    #     model_output_logits = torch.gather(logits_log_softmax, dim=-1, index=tokenized_seqs['input_ids'][:,1:].unsqueeze(-1)).squeeze(-1)
    #     return model_output_logits 