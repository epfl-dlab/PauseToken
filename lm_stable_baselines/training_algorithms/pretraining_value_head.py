from lm_stable_baselines.training_algorithms.abstract_on_policy import AbstractLMOnPolicy
from stable_baselines3.ppo.ppo import PPO
import torch
from stable_baselines3.common.utils import explained_variance
import torch.nn.functional as F
import numpy as np


class PretrainingValueHead(AbstractLMOnPolicy, PPO):
    def __init__(self, *args, loss_computed_in_forward_pass, batch_size, use_base_model_for_learning=False, 
                saving_path=None, **kwargs):
        
        # taking care of ppo arguments
        ppo_kwargs = {k: kwargs[k] for k in kwargs if k in PPO.__init__.__code__.co_varnames}
        PPO.__init__(self, *args, **ppo_kwargs)

        AbstractLMOnPolicy.__init__(self, loss_computed_in_forward_pass=loss_computed_in_forward_pass, 
                                    batch_size=batch_size, 
                                    use_base_model_for_learning=use_base_model_for_learning)
        
        self.policy.lm.requires_grad = False
        self.policy.value_head.requires_grad = True
        if saving_path is not None:
            self.saving_path = saving_path
        else:
            self.saving_path = "/data/value_head/value_head.pth"
    
    def collect_rollouts(self, *args, **kwargs):
        # Override if LM-specific logic is necessary
        return AbstractLMOnPolicy.collect_rollouts(self, *args, **kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        value_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                values_pred = values
                
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                # cross entropy loss
                # value_loss = torch.nn.CrossEntropyLoss()(values_pred, rollout_data.returns)
                value_losses.append(value_loss.item())

                loss = value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

            # Logs
        
            # self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/loss", loss.item())


        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        U = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        # save the lm_head 
        print("Saving the value head at ", self.saving_path)
        torch.save(self.policy.value_head.state_dict(), self.saving_path)
        return U
