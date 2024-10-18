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
from lm_stable_baselines.utils import add_filler_tokens
from copy import deepcopy
from stable_baselines3.common.utils import should_collect_more_steps




class STaR(OffPolicyAlgorithm):
    
    def __init__(self,*args, loss_computed_in_forward_pass ,**kwargs):
        kwargs["support_multi_env"] = True
        super().__init__(*args, **kwargs)
        self._setup_model()
        assert all([isinstance(myenv, LanguageModelEnv) for myenv in self.env.envs]), "All environments must be of type LanguageModelEnv"
        all_filler_token = [myenv.filler_token for myenv in self.env.envs]
        assert all([filler_token == all_filler_token[0] for filler_token in all_filler_token]), "All environments must have the same filler token"
        self.policy.filler_token = all_filler_token[0]
        self.replay_buffer.set_filler_token(all_filler_token[0])
        self.env.set_filler_token(all_filler_token[0])
        self.loss_computed_in_forward_pass = loss_computed_in_forward_pass
        
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
        
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        
        rew_thrsh = replay_buffer.reward_threshold
        
        print("Rollout stage")
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            
            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    if rew_thrsh is not None:
                        above_thrsh = rewards[idx] > rew_thrsh
                        num_collected_episodes += int(above_thrsh)
                        self._episode_num += int(above_thrsh)  
                    else:
                        # Update stats
                        num_collected_episodes += 1
                        self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                        
        callback.on_rollout_end()

        res = RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
        
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
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = add_filler_tokens(next_obs_[key], next_obs[key].shape[1], self.policy.filler_token)
                else:
                    next_obs[i] = add_filler_tokens(infos[i]["terminal_observation"], next_obs.shape[1], self.policy.filler_token)
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
                        
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.train()
        
        self._update_learning_rate(self.policy.optimizer)
        nll_losses = []
        print("Gradient update stage")
        for _ in range(gradient_steps):
            self._n_updates += 1
            
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            labels = replay_data.next_observations["input_ids"] if self.loss_computed_in_forward_pass else None

            output = self.policy(replay_data.next_observations, labels=labels)
            
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