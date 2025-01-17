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

        for _ in range(n_batches):

            self._n_updates += 1
            data = self.rollout_buffer.sample_batch(self.batch_size, env=self._vec_normalize_env)
            next_observation = self.get_next_observation(data)
                    
            if self.loss_computed_in_forward_pass:

                if self.ft_on_action_only:
                    observations = data.observations
                    action_start_indices = (observations['input_ids'] != self.policy.tokenizer.pad_token_id).sum(dim=1)
                    labels = next_observation["input_ids"].clone()
                    for idx in range(labels.size(0)):
                        labels[idx, :action_start_indices[idx]] = -100
                    labels[labels == self.policy.tokenizer.pad_token_id] = -100
                else:
                    labels = next_observation["input_ids"]
                    labels_list = list(labels.cpu())
                    collated_labels = self.data_collator(labels_list)
                    labels = collated_labels["labels"].to(self.device) # check with self.policy.tokenizer.decode(labels[0][labels[0]>0])
            else:
                labels = None
            kwargs = {}

            input_ids = next_observation['input_ids'].to(self.device)
            attention_mask=next_observation['attention_mask'].to(self.device)
            labels = labels.to(self.device)

            output = self.policy.lm(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels if labels is not None else input_ids, #  should not do [:, 1:], it's taken care of.
                **kwargs
            )
            if self.loss_computed_in_forward_pass:
                nll_loss = output.loss
                #if control token model you can also get these losses:
                #control_token_loss = output.ctrl_tok_loss
                #lm_loss = output.lm_loss
            else:
                raise NotImplementedError("To be implemented")
                # nll_loss = self.policy.compute_nll_loss(output.logits, labels)
                
            # getting log_probs for importance sampling
            old_log_probs = data.old_log_prob

            logprobs = torch.log_softmax(output.logits, dim = -1)[:, :-1, :]
            input_ending_ids = (data.observations['input_ids']!=0).sum(dim=-1) - 1
            new_logprobs = self.policy._compute_logprobs(logprobs, input_ids[:, 1:], input_ending_ids)
            ratio = torch.exp(new_logprobs - old_log_probs).detach() 
            ratios.append(ratio.mean().item())
            # Compute the loss
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
             
        self.logger.record("train/nll_loss", np.mean(nll_losses))
        self.logger.record("train/ratio", np.mean(ratios))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        
        