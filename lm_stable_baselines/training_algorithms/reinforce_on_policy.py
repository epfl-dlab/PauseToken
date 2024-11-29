from lm_stable_baselines.training_algorithms import STaROnPolicy
import torch
from torch.nn.functional import nll_loss
import numpy as np
class ReinforceOnPolicy(STaROnPolicy):
    
    def __init__(self, baseline_init_val , baseline_lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline_init_val
        self.baseline_lr = baseline_lr
    
    def update_baseline(self, rewards):
        #running average$
        self.baseline = self.baseline + self.baseline_lr * (rewards - self.baseline).mean()
        
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
    
    def make_labels_for_action_only(self,next_observations: torch.Tensor, actions: torch.Tensor, labels: torch.Tensor):
        """
        Returns the start and end indices of actions in the next_observations tensor.

        Args:
            next_observations (torch.Tensor): Tensor of shape (batch_size, k)
            actions (torch.Tensor): Tensor of shape (batch_size, m)

        Returns:
            torch.Tensor: Start indices of the actions in next_observations.
            torch.Tensor: End indices of the actions in next_observations.
        """
        batch_size, k = next_observations.shape
        mask = torch.ones_like(next_observations, dtype=torch.bool)
     

        for i in range(batch_size):
            obs = next_observations[i]
            act = actions[i]
            # Remove padding tokens
            act = act[act != self.policy.tokenizer.pad_token_id]
            m = len(act)
            found_match = False
            # Find the start index of the sequence in the observation
            for j in range(k - m + 1):  # Slide across the observation
                if torch.equal(obs[j:j + m], act):
                    start_idx = j
                    end_idx = j + m - 1
                    mask[i, start_idx : end_idx] = False
                    found_match = True
                    break
            if not found_match:
                raise ValueError(f"Action {act} not found in ${next_observations[i]}")
        
        labels = torch.where(mask.to(labels.device), -100, labels )
        return labels
    
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
        advantages = []

        self.rollout_buffer.find_where_advantage_exceeds_threshold(self.rollout_buffer.advantages)
        n_batches = self.rollout_buffer.data_size // self.batch_size
        self.policy.tokenizer.padding_side = "right"
        baseline_to_log = self.baseline
        for _ in range(n_batches):
            
            self._n_updates += 1
            data = self.rollout_buffer.sample_batch(self.batch_size, env=self._vec_normalize_env)
            next_observation = self.get_next_observation(data).to(self.device)
            
            labels = next_observation["input_ids"]
            labels_list = list(labels.cpu())
            collated_labels = self.data_collator(labels_list)
            labels = collated_labels["labels"].to(self.device) # check with 
           
            labels = self.make_labels_for_action_only(next_observation["input_ids"], data.actions, labels)

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
            
            if num_above_baseline[-1] == 0:
                above_baseline_nll.append(0)
                above_baseline_rewards.append(0)
            else:
                above_baseline_nll.append(sample_nll[above_baseline_condition].mean().item())
                above_baseline_rewards.append(data.advantages[above_baseline_condition].mean().item())
                
            if num_below_baseline[-1] == 0:
                below_baseline_nll.append(0)
                below_baseline_rewards.append(0)
            else:
                below_baseline_nll.append(sample_nll[below_baseline_condition].mean().item())
                below_baseline_rewards.append(data.advantages[below_baseline_condition].mean().item())
            
            losses.append(loss.item())
            
            nll_list.append(nll.item())
            
            rewards.append(data.advantages.mean().item())
            advantages.append((data.advantages - self.baseline).mean().item())

            self.policy.optimizer.zero_grad()
            
            loss.backward()
            
            self.policy.optimizer.step()
            
            baseline_to_log = self.baseline
            
            self.update_baseline(data.advantages)
                    
        self.logger.record("train/avg_reward", np.mean(rewards))
        self.logger.record("train/reinforce_loss", np.mean(losses))
        self.logger.record("train/mean_nll", np.mean(nll_list))
        self.logger.record("train/mean_advantage", np.mean(advantages))
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