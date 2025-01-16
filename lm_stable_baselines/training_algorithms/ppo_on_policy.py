from lm_stable_baselines.training_algorithms.abstract_on_policy import AbstractLMOnPolicy
from gymnasium import spaces
from stable_baselines3.ppo.ppo import PPO
import torch
import numpy as np
from stable_baselines3.common.utils import explained_variance


class PPOOnPolicy(AbstractLMOnPolicy, PPO):
    def __init__(self, *args, loss_computed_in_forward_pass, batch_size, use_base_model_for_learning=False, **kwargs):
        
        # taking care of ppo arguments
        ppo_kwargs = {k: kwargs[k] for k in kwargs if k in PPO.__init__.__code__.co_varnames}
        PPO.__init__(self, *args, **ppo_kwargs)

        AbstractLMOnPolicy.__init__(self, loss_computed_in_forward_pass=loss_computed_in_forward_pass, 
                                    batch_size=batch_size, 
                                    use_base_model_for_learning=use_base_model_for_learning)
        self.n_grad_accumulation_steps = kwargs.get("n_grad_accumulation_steps", 1)
    
    def collect_rollouts(self, *args, **kwargs):
        # Override if LM-specific logic is necessary
        return AbstractLMOnPolicy.collect_rollouts(self, *args, **kwargs)

    def train(self):
        # Use your custom training logic
        # return PPO.train(self)
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

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        ls_returns = []
        ls_advantages = []
        gradient_accumulation_counter = 0
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                # for obs, act in zip(rollout_data.observations["input_ids"], actions):
                #     print("obs: \n", self.policy.tokenizer.decode(obs, skip_special_tokens=True))
                #     print("act: \n", self.policy.tokenizer.decode(act, skip_special_tokens=True))
                #     print()
                # breakpoint()
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                
                observations = rollout_data.observations

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages

                ls_advantages.append(advantages.mean().item())
                ls_returns.append(rollout_data.returns.mean().item())

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                loss.backward()
                gradient_accumulation_counter += 1
                if gradient_accumulation_counter == self.n_grad_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    self.policy.optimizer.zero_grad()
                    gradient_accumulation_counter = 0
                

            self._n_updates += 1
            if not continue_training:
                break

        if gradient_accumulation_counter != 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()
            gradient_accumulation_counter = 0
        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/advantages", np.mean(ls_advantages))
        self.logger.record("train/returns", np.mean(ls_returns))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

