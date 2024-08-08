from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from lm_stable_baselines.environments import LanguageModelEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Optional, Union, Dict, Any, List
import numpy as np
from lm_stable_baselines.utils import add_filler_tokens

class STaR(OffPolicyAlgorithm):
    
    def __init__(self,*args,**kwargs):
        kwargs["support_multi_env"] = True
        super().__init__(*args, **kwargs)
        self._setup_model()
        assert all([isinstance(myenv, LanguageModelEnv) for myenv in self.env.envs]), "All environments must be of type LanguageModelEnv"
        all_filler_token = [myenv.filler_token for myenv in self.env.envs]
        assert all([filler_token == all_filler_token[0] for filler_token in all_filler_token]), "All environments must have the same filler token"
        self.policy.filler_token = all_filler_token[0]
        self.replay_buffer.set_filler_token(all_filler_token[0])
        self.env.set_filler_token(all_filler_token[0])
        
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
        res = super().collect_rollouts(
            env,
            callback,
            train_freq,
            replay_buffer,
            action_noise,
            learning_starts,
            log_interval
        )
        self.policy.tokenizer.padding_side = og_padding_side
        return res
    
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
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                infos[i]["terminal_observation"] = \
                    add_filler_tokens(
                        infos[i]["terminal_observation"],
                        len(self.observation_space),
                        self.policy.filler_token
                    )
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
                        
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.train()
        
        self._update_learning_rate(self.policy.optimizer)
                
        nll_losses = []
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            output = self.policy(replay_data.observations)
            
            nll_loss = self.policy.compute_nll_loss(output.logits, replay_data.observations)
            
            nll_losses.append(nll_loss.item())
            
            self.policy.optimizer.zero_grad()
            
            nll_loss.backward()
            
            self.policy.optimizer.step()
                    
            
        self.logger.record("train/nll_loss", np.mean(nll_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")