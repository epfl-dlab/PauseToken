from lm_stable_baselines.policies.llm_base_policy_value_model import LLMBasePolicyValueModel
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Type, Optional, Dict, Any, Tuple, List, Callable
from gymnasium import spaces
import torch
import gymnasium as gym
from transformers import PreTrainedModel, GenerationConfig,PreTrainedTokenizer, LogitsProcessorList, StoppingCriteria
from transformers.generation.streamers import BaseStreamer
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor
import warnings
import numpy as np
from lm_stable_baselines.utils import add_filler_tokens
from lm_stable_baselines.utils import remove_filler_tokens
from src.utils.constants import ANSWER_TEMPLATE
from lm_stable_baselines.environments.language_model_env import LanguageModelEnv
from src.model.generation.prefix_allowed_tokens.pause_gt_constraint import PauseGroundTruthConstraint
from src.model.generation.logit_processors.pause_logit_processor import PauseLogitsProcessor

class LLMBasePolicyValueModelGTConstrained(LLMBasePolicyValueModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_template = self.tokenizer.encode(ANSWER_TEMPLATE, return_tensors="pt").squeeze()
        self.tokens_to_filter = self.lm.config.control_token_to_id.values()
        
        self.allowed_tokens_to_predict = list(self.tokens_to_filter)
        self.tokens_to_filter = list(self.tokens_to_filter)
        self.tokens_to_filter.append(self.tokenizer.pad_token_id)


    def _predict(self, observation: PyTorchObs, deterministic: bool = False, return_dict: bool = False):
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
          
        
        was_in_training = self.lm.training
        assert self.generation_params_to_use is not None, \
            "You've never set the generation config to use. Please set it using the set_generation_cfg method. Options are 'train' or 'test'"
        generation_params = self.generation_params[self.generation_params_to_use]

        self.lm.eval()
        og_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        feature = self.extract_features(observation)
        inputs = feature["input_ids"]
        self.pre_predict(feature)

        if not self.use_peft_at_inference:
            self.lm.disable_adapter_layers()

        if self.generation_params_to_use == "train":
            
            constraint_module = PauseGroundTruthConstraint(
                tokens_to_filter=self.tokens_to_filter,
                max_tokens=generation_params["generation_config"].max_length,
                allowed_tokens_to_predict=self.allowed_tokens_to_predict
            )
            #kind of hack to get pause token id
            pause_logit_processor = PauseLogitsProcessor(
                self.allowed_tokens_to_predict[0],
                constraint_module=constraint_module,
            )

            gt_array_cp =  torch.from_numpy(LanguageModelEnv.gt_array).clone()
            gt_array_cp[gt_array_cp == -100] = self.tokenizer.pad_token_id

            prefix_allowed_tokens_fn = pause_logit_processor.set_prefix_allowed_tokens_fn(
                batch_info={
                    "tokenized_ground_truths": gt_array_cp,
                    "pad_token_id": self.tokenizer.pad_token_id
                }
            )
            if "prefix_allowed_tokens_fn" in generation_params:
                generation_params.pop("prefix_allowed_tokens_fn")
                
            if "logits_processor" in generation_params:
                generation_params.pop("logits_processor")
        

            with torch.no_grad():
                outputs = self.lm.generate(
                    inputs = inputs,
                    attention_mask = feature["attention_mask"],
                    tokenizer=self.tokenizer,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    logits_processor=LogitsProcessorList([pause_logit_processor]),
                    **generation_params,
                )
        else:
            with torch.no_grad():
                outputs = self.lm.generate(
                    inputs = inputs,
                    attention_mask = feature["attention_mask"],
                    tokenizer=self.tokenizer,
                    **generation_params,
                )
            
        if not self.use_peft_at_inference:
            self.lm.enable_adapter_layers()
        
        outputs =  self.post_predict(inputs, outputs, return_dict = return_dict)
        if was_in_training:
            self.lm.train()
        self.tokenizer.padding_side = og_padding_side  
        return outputs