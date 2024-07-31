import re
import numpy as np
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

def strip_special_tokens(text,tokenizer):
    """ Strip special tokens from a text
    
    :param text: Text to strip special tokens from
    :type text: str
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :return: Text without special tokens
    :rtype: str
    """
    tokenized_text = tokenizer(text)["input_ids"]
    return tokenizer.decode(tokenized_text, skip_special_tokens=True)



import sys
from typing import List,Dict,Union
import torch


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
    def __init__(self, tokenizer):
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
    def __init__(self, tokenizer, model,tokens_ids_to_ignore,invalid_answer_penalty = torch.finfo(torch.float).min, check_correctess = False):
        self.tokenizer = tokenizer
        self.model = model
        self.tokens_ids_to_ignore = tokens_ids_to_ignore
        self.invalid_ans_penalty = invalid_answer_penalty
        self.check_correctess = check_correctess
        if check_correctess:
            self.correctness_reward = GSM8KCorrectnessReward(tokenizer)
        
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
        
        is_correct = True
        if self.check_correctess:
            is_correct = (self.correctness_reward(model_output,ground_truth) == 1)
        
        
        answer = extract_answer(strip_special_tokens(model_output,self.tokenizer))
        if answer == INVALID_ANS or not is_correct:
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


class GSM8KFinalAnswerLogLikelihoodReward(LogLikelihoodReward):
    def __init__(self, *args, delimiter= "####", **kwargs):
        super().__init__(*args,**kwargs)
        self.tokens_ids_to_ignore.append(self.tokenizer.eos_token_id)
        self.tokens_ids_to_ignore.append(self.tokenizer.encode(' ')[-1])
        self.identifiers = self.find_all_token_ids(delimiter)
        
    def find_all_token_ids(self,string: str):
        #iterate through tokenizer tokens
        matching_token_ids = []
        for token,id in self.tokenizer.get_vocab().items():
            if string in token[-len(string):]:
                matching_token_ids.append(id)
                
        return torch.tensor(matching_token_ids)

    def get_start_of_answer_token_position(self, model_output: str):
        #find the position of the first token of the answer
        tokenized_output = self.tokenizer(model_output, return_tensors="pt", padding=False, truncation=True)
        start_of_answer_token_position = torch.where(torch.isin(tokenized_output['input_ids'], self.identifiers))[1][-1]
        return start_of_answer_token_position

    def get_masked_output_logits(self, model_output: str, padding: bool):
        tokenized_seqs = self.tokenizer(model_output, return_tensors="pt", padding=padding, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**tokenized_seqs, return_dict=True)

        # discard the last token, it's model's output to EOS
        # logits_log_softmax = torch.nn.functional.log_softmsax(outputs.logits[:,:-1,:], dim=-1)
 
        logits_log_softmax = torch.nn.functional.log_softmax(outputs.logits[:, :-1, :], dim=-1)
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
    
    def batch_call(self, model_output: List[str], ground_truth: List[str]):
        raise NotImplementedError("batch_call not implemented for GSM8KFinalAnswerLogLikelihoodReward would need to fix get_start_of_answer_token_position")
    
    
from gymnasium import Env
from typing import List
from transformers import PreTrainedTokenizer
from datasets import Dataset
class myEnv(Env):
    
    def __init__(
        self,
        reward: AbstractReward,
        tokenizer: PreTrainedTokenizer,
        termination_tokens: List[int],
        max_tokens: int,
    ):
        self.reward = reward
        self.termination_tokens = termination_tokens
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.observation_space =  spaces.Discrete(tokenizer.vocab_size)
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        self.current_state = []
    def step(self, action):
        self.current_state.append(action.item())
        return self._get_obs()

    def is_terminated(self, state: List[int]):
        return any([token in state[1:] for token in self.termination_tokens])
    
    def is_truncated(self, state: List[int]):
        if not self.is_terminated(state) and len(state) >= self.max_tokens:
            return True
        return False

    def reset(
        self,
        seed = 123,
        options = None,
    ):  # type: ignore
        
        super().reset(seed=seed)
        self.current_state = [self.tokenizer.bos_token_id]
        self.last_obs = self.tokenizer.bos_token_id
        self.terminated = False
        self.truncated = False
        self.done = False
        return self.last_obs, {}
    
    def _get_obs(self):
        decode_state = self.tokenizer.decode(self.current_state)
        reward = self.reward(decode_state, "#### 1")
        is_terminated =  self.is_terminated(self.current_state)
        is_truncated = self.is_truncated(self.current_state)
        info = {}
        self.last_obs = self.current_state[-1]
        return self.last_obs , reward, is_terminated, is_truncated, info

    def render(self):
        breakpoint()
        return self.tokenizer.decode(self.current_state)

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        pass
    
    
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Type, Optional, Dict, Any
from gymnasium import spaces
import torch
import gymnasium as gym
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_linear_fn
class LLMBasePolicy(BasePolicy):
    
    def __init__(
        self,
        *args,
        lr_schedule: Schedule,
        lm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        squash_output: bool = False,
        **kwargs,
    ):
        
        if observation_space is None:
            #check vocab size
            observation_space = spaces.Discrete(lm.config.vocab_size)
        if action_space is None:
            action_space = spaces.Discrete(lm.config.vocab_size)
        
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        self.lm = lm
        self.tokenizer = tokenizer
        self._build(lr_schedule)
        
    def _build(self, lr_schedule: Schedule):
        self.optimizer = self.optimizer_class(
            self.lm.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.lm.eval()
        
    
    def compute_nll_loss(self, logits, labels):
        shift_lm_logits = logits[..., :-1, :].contiguous()
        shift_lm_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_lm_logits = shift_lm_logits.view(-1, self.lm.config.vocab_size)
        shift_lm_labels = shift_lm_labels.view(-1)
        # Ensure tensors are on the same device
        shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(shift_lm_logits, shift_lm_labels)
    
    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PyTorchObs:
        
        if (isinstance(obs, dict) and 'input_ids' in obs and 'attention_mask' not in obs):
            obs = obs["input_ids"]
        #make input_ids and attention_mask
        if isinstance(obs, torch.Tensor):
            decoded_seq = self.tokenizer.batch_decode(obs)
            feature = self.tokenizer(decoded_seq, return_tensors="pt", padding=True, truncation=True)
        elif isinstance(obs, dict):
            feature = obs
        else:
            raise ValueError("Observation type not supported")
        return feature
            
            
        
    def forward(self, obs: PyTorchObs, action = None) -> torch.Tensor:
        
        feature = self.extract_features(obs)
        feature = {k: v.to(self.device) for k, v in feature.items()}
        return self.lm(**feature)
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False):
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        output = self.forward(observation)
        logits = output.logits[...,-1,:]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if deterministic:
            return probs.argmax(dim=-1)
        else:
            return torch.multinomial(probs, num_samples=1)

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
class STaR(OffPolicyAlgorithm):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_model()
    
    
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.train()
        
        self._update_learning_rate(self.policy.optimizer)
                
        nll_losses = []
        
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            breakpoint()
            
            actions = replay_data.actions
            
            output = self.policy(replay_data.observations)
            
            nll_loss = self.policy.compute_nll_loss(output.logits, actions)
            
            nll_losses.append(nll_loss.item())
            
            self.policy.optimizer.zero_grad()
            
            nll_loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
                    
            
        self.logger.record("train/nll_loss", np.mean(nll_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        

        

if __name__ == "__main__":
    #load gpt2 model
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
    from stable_baselines3.common.buffers import ReplayBuffer 
    lm = GPT2LMHeadModel.from_pretrained("gpt2", device_map = "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.unk_token
    #get eos token id
    eos_token_id = tokenizer.eos_token_id
    #initialize reward
    reward = GSM8KFinalAnswerLogLikelihoodReward(tokenizer,lm,[])

    env = myEnv(reward,tokenizer, [eos_token_id], 1024)
    
    tokenizer.pad_token = tokenizer.unk_token
    obs,_ = env.reset()
    model = LLMBasePolicy(lr_schedule = get_linear_fn(start = 1e-5, end = 1e-7, end_fraction= 0.95), lm=lm, tokenizer=tokenizer)
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, is_truncated, info = env.step(action)
        if done:
            break
    print("Result of Prediction")
    print(env.render()) 
    
    # policy = LLMBasePolicy
    # policy_kwargs = {"lm": model, "tokenizer": tokenizer, "lr_schedule": get_linear_fn(start = 1e-5, end = 1e-7, end_fraction= 0.95)}
    
    # algo = STaR(
    #     policy = policy,
    #     policy_kwargs = policy_kwargs,
    #     env=env,
    #     learning_rate=1e-5,
    #     train_freq = TrainFreq(5, TrainFrequencyUnit.STEP),replay_buffer_class= ReplayBuffer
    # )
    # algo.learn(1000)
     