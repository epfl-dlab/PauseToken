from lm_stable_baselines.training_algorithms.abstract_on_policy import AbstractLMOnPolicy
from stable_baselines3.ppo.ppo import PPO
import torch
import numpy as np


class PPOOnPolicy(AbstractLMOnPolicy, PPO):
    def __init__(self, *args, loss_computed_in_forward_pass, batch_size, use_base_model_for_learning=False, **kwargs):
        
        # taking care of ppo arguments
        ppo_kwargs = {k: kwargs[k] for k in kwargs if k in PPO.__init__.__code__.co_varnames}
        PPO.__init__(self, *args, **ppo_kwargs)

        AbstractLMOnPolicy.__init__(self, loss_computed_in_forward_pass=loss_computed_in_forward_pass, 
                                    batch_size=batch_size, 
                                    use_base_model_for_learning=use_base_model_for_learning)
        
    
    def collect_rollouts(self, *args, **kwargs):
        # Override if LM-specific logic is necessary
        return AbstractLMOnPolicy.collect_rollouts(self, *args, **kwargs)

    def train(self):
        # Use your custom training logic
        return PPO.train(self)
