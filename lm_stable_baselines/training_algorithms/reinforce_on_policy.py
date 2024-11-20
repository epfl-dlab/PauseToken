from lm_stable_baselines.training_algorithms import STaROnPolicy
import torch
from torch.nn.functional import nll_loss
import numpy as np
class ReinforceOnPolicy(STaROnPolicy):
    
    def __init__(self, baseline_lr, baseline_init_val, token_level_actions , discount_factor ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples = 0
        self.baseline = baseline_init_val
    
    def update_baseline(self, rewards):
        #running average$
        num_rewards = len(rewards)
        self.baseline = (self.baseline  * self.seen_samples  + rewards.sum()) / (self.seen_samples + num_rewards)
        self.seen_samples += num_rewards
        
    def token_level_reinforce_loss(
        lm_output,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        rewards: torch.FloatTensor,  
    ):
        raise NotImplementedError("Token level reinforce loss not implemented")
    
    def reinforce_loss(
        self,
        lm_output,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        rewards: torch.FloatTensor,         
    ):

        if hasattr(self.policy.lm, "compute_loss"):
            
            nll, _ , _ = self.policy.lm.compute_loss(
                labels,
                lm_logits=lm_output.lm_logits,
                ctrl_tok_logits=lm_output.control_token_logits,
                attention_mask=attention_mask,
                reduce_mean=False,
            )
            advantage = (rewards - self.baseline)
            policy_losses =  advantage.detach() * nll
            policy_loss = policy_losses.mean()
        
        else:
            raise NotImplementedError("Reinforce loss not implemented for non control token models Yet")
            # nll = nll_loss(
            #     lm_output.logits,
            #     labels,
            #     reduce=False,
            #     reduction=None,
            # )
            #~~~ NLL loss ~~~~~
        
        # policy_loss = torch.cat(policy_loss).to(self.device).mean()
        return policy_loss, nll.mean() , nll
            
    
    def train(self) -> None:
        
        self.policy.train()
        
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        rewards = []
        nll_list = []
        above_baseline_nll = []
        below_baseline_nll = []
        above_baseline_rewards = []
        below_baseline_rewards = []
        num_above_baseline = []
        num_below_baseline = []

        self.rollout_buffer.find_where_advantage_exceeds_threshold(self.rollout_buffer.advantages)
        n_batches = self.rollout_buffer.data_size // self.batch_size
        self.policy.tokenizer.padding_side = "right"
        for _ in range(n_batches):
            
            self._n_updates += 1
            data = self.rollout_buffer.sample_batch(self.batch_size, env=self._vec_normalize_env)
            next_observation = self.get_next_observation(data)
            
            labels = next_observation["input_ids"]
            labels_list = list(labels.cpu())
            collated_labels = self.data_collator(labels_list)
            labels = collated_labels["labels"].to(self.device) # check with self.policy.tokenizer.decode(labels[0][labels[0]>0])

            input_ids = next_observation["input_ids"].to(self.device)
            attention_mask = next_observation["attention_mask"].to(self.device)
            
            output = self.policy.lm(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels=labels if self.loss_computed_in_forward_pass else None
            )
            
            loss, nll, sample_nll = self.reinforce_loss(
                output,
                labels,
                attention_mask,
                data.advantages
            )
            
            above_baseline_condition = data.advantages >= self.baseline
            below_baseline_condition = data.advantages < self.baseline
            
            num_above_baseline.append(above_baseline_condition.sum().item())
            num_below_baseline.append(below_baseline_condition.sum().item())
            
            above_baseline_nll.append(sample_nll[above_baseline_condition].mean().item())
            below_baseline_nll.append(sample_nll[below_baseline_condition].mean().item())
            above_baseline_rewards.append(data.advantages[above_baseline_condition].mean().item())
            below_baseline_rewards.append(data.advantages[below_baseline_condition].mean().item())
            
            losses.append(loss.item())
            
            nll_list.append(nll.item())
            
            rewards.append(data.advantages.mean().item())

            self.policy.optimizer.zero_grad()
            
            loss.backward()
            
            self.policy.optimizer.step()
            
            baseline_to_log = self.baseline
            
            self.update_baseline(data.advantages)
                    
        self.logger.record("train/avg_reward", np.mean(rewards))
        self.logger.record("train/reinforce_loss", np.mean(losses))
        self.logger.record("train/mean_nll", np.mean(nll_list))
        
        mean_above_baseline_nll = (np.asarray(num_above_baseline) * np.asarray(above_baseline_nll)).sum() / np.sum(num_above_baseline)
        mean_below_baseline_nll = (np.asarray(num_below_baseline) * np.asarray(below_baseline_nll)).sum() / np.sum(num_below_baseline)
        mean_above_baseline_rewards = (np.asarray(num_above_baseline) * np.asarray(above_baseline_rewards)).sum() / np.sum(num_above_baseline)
        mean_below_baseline_rewards = (np.asarray(num_below_baseline) * np.asarray(below_baseline_rewards)).sum() / np.sum(num_below_baseline)
        self.logger.record("train/above_baseline_nll", mean_above_baseline_nll)
        self.logger.record("train/below_baseline_nll", mean_below_baseline_nll)
        self.logger.record("train/above_baseline_rewards", mean_above_baseline_rewards)
        self.logger.record("train/below_baseline_rewards", mean_below_baseline_rewards)
        self.logger.record("train/baseline", baseline_to_log)
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")