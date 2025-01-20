from lm_stable_baselines.training_algorithms.abstract_on_policy import AbstractLMOnPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import torch
import numpy as np



class STaROnPolicy(AbstractLMOnPolicy,OnPolicyAlgorithm):
    
    def __init__(self,*args, loss_computed_in_forward_pass=True, batch_size=8, use_base_model_for_learning=False, ft_on_action_only=False ,**kwargs):

        # taking care of on policy arguments
        on_policy_kwargs = {k: kwargs[k] for k in kwargs if k in OnPolicyAlgorithm.__init__.__code__.co_varnames}
        OnPolicyAlgorithm.__init__(self, *args, **on_policy_kwargs)

        AbstractLMOnPolicy.__init__(self, loss_computed_in_forward_pass=loss_computed_in_forward_pass, 
                         batch_size=batch_size, use_base_model_for_learning=use_base_model_for_learning)
        
        self.ft_on_action_only = ft_on_action_only
        self.n_grad_accumulation_steps = kwargs.get("n_grad_accumulation_steps", 1)

    def train(self) -> None:
        
        if hasattr(self.policy.lm, "enable_adapter_layers"):
            self.policy.lm.enable_adapter_layers()

        if "peft_to_train" in self.name_to_adapter:
            self.policy.lm.set_adapter(self.name_to_adapter["peft_to_train"])
        
        self.policy.train()
        
        self._update_learning_rate(self.policy.optimizer)
        nll_losses, ratios = [], []

        self.rollout_buffer.find_where_advantage_exceeds_threshold(self.rollout_buffer.advantages)
        n_batches = self.rollout_buffer.data_size // self.batch_size + (self.rollout_buffer.data_size % self.batch_size != 0)
        self.policy.tokenizer.padding_side = "right"
        gradient_accumulation_counter = 0
        self.prev_n_updates = self._n_updates
        for _ in range(n_batches):

            self._n_updates += 1
            data = self.rollout_buffer.sample_batch(self.batch_size, env=self._vec_normalize_env)

            logprobs = self.policy.evaluate_actions(data.observations, data.actions)[1]
            
            ratio = torch.exp(logprobs - data.old_log_prob).detach() 
            ratios.append(ratio.mean().item())
            # Compute the loss
            nll_loss = -logprobs.mean()
            nll_losses.append(nll_loss.item())
            self.policy.optimizer.zero_grad()

            nll_loss.backward()

            gradient_accumulation_counter += 1
            if gradient_accumulation_counter == self.n_grad_accumulation_steps:
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()
                gradient_accumulation_counter = 0
        
        if gradient_accumulation_counter != 0:
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()
            gradient_accumulation_counter = 0
            # self.update_baseline(gradient_accumulation_counter)
        
        if n_batches > 0:
            self.logger.record("train/nll_loss", np.mean(nll_losses))
            self.logger.record("train/ratio", np.mean(ratios))
            self.logger.record("train/n_updates", self._n_updates - self.prev_n_updates, exclude="tensorboard")
        
        