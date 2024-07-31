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
        filler_token: int = -100,
    ):
        self.reward = reward
        self.termination_tokens = termination_tokens
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.observation_space =  spaces.MultiDiscrete([tokenizer.vocab_size]* max_tokens, dtype = np.int64) 
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        self.current_state = []
        self.filler_token = filler_token
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
        full_current_state = self.current_state
        if len(self.current_state) < self.max_tokens:
            full_current_state = self.current_state + [self.filler_token] * (self.max_tokens - len(self.current_state))
        return full_current_state , reward, is_terminated, is_truncated, info

    def render(self):
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
from typing import Type, Optional, Dict, Any, Tuple
from gymnasium import spaces
import torch
import gymnasium as gym
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_linear_fn,obs_as_tensor
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
        filler_token: int = -100,
        **kwargs,
    ):
        
        if observation_space is None:
            #check vocab size
            observation_space = spaces.MultiDiscrete([lm.config.vocab_size]* tokenizer.vocab_size, dtype = np.int64) 
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
        self.filler_token = filler_token
        self._build(lr_schedule)
        
    def _build(self, lr_schedule: Schedule):
        self.optimizer = self.optimizer_class(
            self.lm.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.lm.eval()
    
    def obs_to_tensor(self, observation: np.ndarray) -> Tuple[PyTorchObs, bool]:    
        assert isinstance(observation, np.ndarray), "Observation must be a numpy array"
        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, True
        
    def compute_nll_loss(self, logits, labels):
                
        shift_lm_logits = logits[..., :-1, :].contiguous()
        shift_lm_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_lm_logits = shift_lm_logits.view(-1, self.lm.config.vocab_size)
        shift_lm_labels = shift_lm_labels.view(-1)
        # Ensure tensors are on the same device
        shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.filler_token)
        return loss_fct(shift_lm_logits, shift_lm_labels)
    
    def remove_filler_tokens(self, obs: torch.Tensor):
        #check for any filler tokens
        if not (obs == self.filler_token).any():
            return obs
        
        shape = obs.shape
        if len(shape) == 1:
            return obs[obs != self.filler_token].reshape(-1,1)
        else:
            return [ob[ob != self.filler_token] for ob in obs]
            #find 
    
    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PyTorchObs:
        if (isinstance(obs, dict) and 'input_ids' in obs and 'attention_mask' not in obs):
            obs = obs["input_ids"]
        #make input_ids and attention_mask
        if isinstance(obs, torch.Tensor):
            filt_obs = self.remove_filler_tokens(obs)
            decoded_seq = self.tokenizer.batch_decode(filt_obs)
            feature = self.tokenizer(decoded_seq, return_tensors="pt", padding=True, truncation=True)
        elif isinstance(obs, dict) and "input_ids" in obs and "attention_mask" in obs:
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
from stable_baselines3.common.vec_env import VecNormalize 
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer
class LMReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        *args,
        filler_token = -100,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filler_token = filler_token
        self.eos_token_id = eos_token_id
        self.observations.fill(self.filler_token)
        self.actions.fill(self.filler_token)
        if self.optimize_memory_usage:
            self.next_observations.fill(self.filler_token)
    
    def set_filler_token(self, filler_token):
        self.observations[self.observations == self.filler_token] = filler_token
        self.actions[self.actions == self.filler_token] = filler_token
        if self.optimize_memory_usage:
            self.next_observations[self.next_observations == self.filler_token] = filler_token
        self.filler_token = filler_token
        
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        #find sequences where done is True
        done_indices = np.where(done)[0]
        if len(done_indices) > 0:
            finished_obs = obs[done_indices]
            finished_actions = action[done_indices]
            finished_rewards = reward[done_indices]
            finished_infos = [infos[i] for i in done_indices]
            finished_next_obs = next_obs[done_indices]
        
            super().add(finished_obs, finished_next_obs, finished_actions, finished_rewards, done, finished_infos)
            
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.callbacks import BaseCallback
class STaR(OffPolicyAlgorithm):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_model()
        self.policy.filler_token = kwargs["env"].filler_token
        self.replay_buffer.set_filler_token(kwargs["env"].filler_token)
        
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        
        og_padding_side = self.policy.tokenizer.padding_side
        self.policy.tokenizer.padding_side = "left"
        res = super().collect_rollouts(env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval)
        self.policy.tokenizer.padding_side = og_padding_side
        return res
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.train()
        
        self._update_learning_rate(self.policy.optimizer)
                
        nll_losses = []
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            actions = replay_data.actions

            output = self.policy(replay_data.observations)
            
            nll_loss = self.policy.compute_nll_loss(output.logits, replay_data.observations)
            
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
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map = "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.unk_token
    #get eos token id
    eos_token_id = tokenizer.eos_token_id
    #initialize reward
    reward = GSM8KFinalAnswerLogLikelihoodReward(tokenizer,model,[])

    env = myEnv(reward,tokenizer, [eos_token_id], 1024)
    
    
    policy = LLMBasePolicy
    policy_kwargs = {"lm": model, "tokenizer": tokenizer, "lr_schedule": get_linear_fn(start = 1e-5, end = 1e-7, end_fraction= 0.95)}
    
    algo = STaR(
        policy = policy,
        policy_kwargs = policy_kwargs,
        env=env,
        learning_rate=1e-5,
        train_freq = TrainFreq(1, TrainFrequencyUnit.EPISODE),
        replay_buffer_class= LMReplayBuffer,
    )
    algo.learn(1000)
    # tokenizer.pad_token = tokenizer.unk_token
    # initial_state = tokenizer("The dog is")["input_ids"]
    # obs = env.reset(initial_state)
    
    # for i in range(10):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, is_truncated, info = env.step(action)
    #     if done:
    #         break
    # print(env.render())   