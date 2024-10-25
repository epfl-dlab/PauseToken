from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from lm_stable_baselines.environments import LanguageModelEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Optional, Union, Dict, Any, List
import numpy as np
from lm_stable_baselines.utils import add_filler_tokens, remove_filler_tokens
from copy import deepcopy
from stable_baselines3.common.utils import should_collect_more_steps
from lm_stable_baselines.training_algorithms import STaROffPolicy
from src.utils.constants import FEEDBACK_TEMPLATE, ANSWER_TEMPLATE, CORRECT_ANSWER_FEEDBACK, INCORRECT_ANSWER_FEEDBACK


class ConstRCOffPolicy(STaROffPolicy):
    
    def __init__(self,*args ,**kwargs):
        kwargs["support_multi_env"] = True
        super().__init__(*args, **kwargs)
        self._setup_model()
        assert all([isinstance(myenv, LanguageModelEnv) for myenv in self.env.envs]), "All environments must be of type LanguageModelEnv"
        all_filler_token = [myenv.filler_token for myenv in self.env.envs]
        assert all([filler_token == all_filler_token[0] for filler_token in all_filler_token]), "All environments must have the same filler token"
        self.policy.filler_token = all_filler_token[0]
        self.replay_buffer.set_filler_token(all_filler_token[0])
        self.env.set_filler_token(all_filler_token[0])
                
    
    def recondition_terminal_observations(self, infos: List[Dict[str, Any]], rewards:  np.ndarray, dones: np.ndarray) -> None:
        assert "terminal_observation" in infos[0], "The terminal observation must be stored in the info dict"
        decoded_terminal_observations = self.policy.tokenizer.batch_decode([remove_filler_tokens(info["terminal_observation"], self.policy.filler_token)[:,0].tolist() for info in infos])
        for i, info in enumerate(infos):
            if dones[i]:
                decoded_terminal_obs = decoded_terminal_observations[i]
                reward = rewards[i]
                if reward > self.replay_buffer.advantage_threshold:
                    feedback = CORRECT_ANSWER_FEEDBACK
                else:
                    feedback = INCORRECT_ANSWER_FEEDBACK
                
                decoded_terminal_obs = decoded_terminal_obs.replace(CORRECT_ANSWER_FEEDBACK, feedback)
                original_size = info["terminal_observation"].shape[0]
                #bos is already there so we remove the first token (which is also bos)
                info["terminal_observation"] = np.squeeze(
                    add_filler_tokens(self.policy.tokenizer(decoded_terminal_obs, return_tensors="np")["input_ids"][:,1:], original_size, self.policy.filler_token)
                )
        
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        self.recondition_terminal_observations(infos, reward, dones)
        
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
    
    def get_next_observation(self, data):
        return data.next_observations
        
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.train()
        self.replay_buffer.find_where_advantage_exceeds_threshold(self.replay_buffer.rewards, override_advantage_threshold=-np.inf)
        n_batches = self.replay_buffer.data_size // self.batch_size
        self._update_learning_rate(self.policy.optimizer)
        nll_losses = []
        print("Gradient update stage")

        for _ in range(n_batches):
            self._n_updates += 1
            
            replay_data = self.replay_buffer.sample_batch(batch_size, env=self._vec_normalize_env)

            if self.loss_computed_in_forward_pass:
                labels = replay_data.next_observations["input_ids"]
                labels_list = list(labels.cpu())
                collated_labels = self.data_collator(labels_list)
                labels = collated_labels["labels"] 
            else:
                labels = None

            output = self.policy.lm(
                input_ids = replay_data.next_observations["input_ids"],
                attention_mask = replay_data.next_observations["attention_mask"],
                labels=labels.to(self.device)
            )
            
            if self.loss_computed_in_forward_pass:
                nll_loss = output.loss
            else:
                nll_loss = self.policy.compute_nll_loss(output.logits, replay_data.next_observations)
            
            nll_losses.append(nll_loss.item())
            
            self.policy.optimizer.zero_grad()
            
            nll_loss.backward()
            
            self.policy.optimizer.step()
                    
            
        self.logger.record("train/nll_loss", np.mean(nll_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")