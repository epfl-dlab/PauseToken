from lm_stable_baselines.training_algorithms.abstract_on_policy import AbstractLMOnPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium import spaces
import torch
import numpy as np
# from torchviz import make_dot

class ReinforceOnPolicy(AbstractLMOnPolicy, OnPolicyAlgorithm):

    def __init__(self, *args, loss_computed_in_forward_pass=True, batch_size=8, 
                 baseline_init_val=0.0 , baseline_lr=0.01, n_grad_accumulation_steps = 1,
                 use_base_model_for_learning=False, **kwargs):
        # Sutton and Barto, 1998, Page 330

        # taking care of on policy arguments
        on_policy_kwargs = {k: kwargs[k] for k in kwargs if k in OnPolicyAlgorithm.__init__.__code__.co_varnames}
        OnPolicyAlgorithm.__init__(self, *args, **on_policy_kwargs)

        AbstractLMOnPolicy.__init__(self, loss_computed_in_forward_pass=loss_computed_in_forward_pass, 
                         batch_size=batch_size, use_base_model_for_learning=use_base_model_for_learning)
        
        self.n_grad_accumulation_steps = n_grad_accumulation_steps
        self.normalize_advantage = kwargs.get("normalize_advantage", False)
    
    def train(self) -> None:
        
        if hasattr(self.policy.lm, "enable_adapter_layers"):
            self.policy.lm.enable_adapter_layers()
        if "peft_to_train" in self.name_to_adapter:
            self.policy.lm.set_adapter(self.name_to_adapter["peft_to_train"])
        
        self.policy.train()
        self._update_learning_rate(self.policy.optimizer)
        pg_losses = []
        ratios = []
        nll_list = []
        ls_returns = []
        ls_advantages = []
        value_losses = []
        entropy_losses = []

        self.rollout_buffer.find_where_advantage_exceeds_threshold(self.rollout_buffer.advantages)
        n_batches = self.rollout_buffer.data_size // self.batch_size + (self.rollout_buffer.data_size % self.batch_size != 0)
       
        self.policy.tokenizer.padding_side = "right"
        
        gradient_accumulation_counter = 0
        
        for _ in range(n_batches):
            # get the data from the buffer
            data = self.rollout_buffer.sample_batch(self.batch_size, env=self._vec_normalize_env)
            actions = data.actions
            observations = data.observations
            # forward pass through the model
            values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
            ratio = torch.exp(log_prob - data.old_log_prob).detach() 
            ratios.append(ratio.mean().item())
            # Compute the loss
            nll = -log_prob.mean()
            nll_list.append(nll.item())
            # compute the baseline loss (aka the value function)
            values = values.flatten()
            # Normalize advantage
            advantages = data.advantages
            
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pg_loss = (nll * advantages).mean()

            # Logging
            pg_losses.append(pg_loss.item())
            ls_advantages.append(advantages.mean().item())
            ls_returns.append(data.returns.mean().item())

            values_pred = values
            value_loss = torch.nn.functional.mse_loss(data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            self.policy.optimizer.zero_grad()
            loss.backward()

            gradient_accumulation_counter += 1
            if gradient_accumulation_counter == self.n_grad_accumulation_steps:
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()
                gradient_accumulation_counter = 0
        
        if gradient_accumulation_counter != 0:
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()
            gradient_accumulation_counter = 0


        if n_batches > 0:
            self.logger.record("train/pg_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/return", np.mean(ls_returns))
            self.logger.record("train/mean_nll", np.mean(nll_list))
            self.logger.record("train/mean_advantage", np.mean(ls_advantages))
            self.logger.record("train/mean_ratio", np.mean(ratios))


    