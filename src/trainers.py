import torch
from torch import nn
from trl import SFTTrainer
import torch
import warnings
import numpy as np
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import Trainer
from typing import Dict, Union, Any

class SFTIgnoreTokensInLossTrainer(SFTTrainer):
    """ Trainer typically used to pretrain Pause Tokens model. It ignores the loss for the tokens specified in ignore_tokens.
    
    :param ignore_tokens: List of tokens to ignore in the loss computation
    :type ignore_tokens: List[int]
    """
    def __init__(self, ignore_tokens, *args, **kwargs):
        self.ignore_tokens = ignore_tokens
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by SFTTrainerMaskedPauseTokens. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """   
        # Ignore tokens in the loss
        # torch.nn.CrossEntropyLoss ignores tokens with labels -100    
        if "labels" in inputs:
            labels = inputs.pop("labels")
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for token_id in self.ignore_tokens:
                token_mask = (labels == token_id)
                mask[token_mask] = True
            labels[mask] = -100 #ignore index in CrossEntropyLoss
            inputs["labels"] = labels
        
        return super().compute_loss(model, inputs, return_outputs)
       
        


class WeightedSFTTrainer(SFTTrainer):
    """ Weighted SFTTrainer. It computes the loss for each sample in the prefix and then computes the weighted average of the losses (weights based on rewards).
    Note that its not a perfect implementation but it works for now.
    
    :param reward_col_name: Column name in the dataset which contains the rewards
    :type reward_col_name: str
    :param n_samples_per_prefix: Number of samples per prefix (number of y_i for a given x)
    :type n_samples_per_prefix: int
    :param beta: Beta value for softmax
    :type beta: float
    """
    
    def __init__(self, reward_col_name , n_samples_per_prefix, beta ,*args, **kwargs):
        self.reward_col_name = reward_col_name
        super().__init__(*args, **kwargs)
        self._signature_columns = ["input_ids", "attention_mask", self.reward_col_name]
        self.n_samples_per_prefix  = n_samples_per_prefix
        assert beta > 0, "Beta should be greater than 0"
        self.inverse_beta = 1/beta

    def _prepare_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        raise NotImplementedError("Packed dataloader not implemented yet")
    
    def _prepare_non_packed_dataloader(
            self,
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func=None,
            add_special_tokens=True,
            remove_unused_columns=True,
        ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False
                
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            if isinstance(element[dataset_text_field][0],list):
                outputs_input_ids = []
                outputs_attention_mask = []
                for el in element[dataset_text_field]:
                    outputs = tokenizer(
                        el if not use_formatting_func else formatting_func(element),
                        add_special_tokens=add_special_tokens,
                        truncation=True,
                        padding=True,
                        max_length=max_seq_length,
                        return_overflowing_tokens=False,
                        return_length=False,
                    )
                    outputs_input_ids.append(outputs["input_ids"])
                    outputs_attention_mask.append(outputs["attention_mask"])
                return {"input_ids": outputs_input_ids, "attention_mask": outputs_attention_mask}
            
            else:
                outputs = tokenizer(
                    element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )

            # if use_formatting_func and not self._dataset_sanity_checked:
            #     if not isinstance(formatting_func(element), list):
            #         raise ValueError(
            #             "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
            #         )
            #     else:
            #         self._dataset_sanity_checked = True

                return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
                
        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        tokenized_dataset = tokenized_dataset.add_column(self.reward_col_name, dataset[self.reward_col_name])
        return tokenized_dataset
        
    def compute_weights(self, inputs):
        device = inputs["input_ids"].device
        #assuming all inputs come from the same x here
        rewards = inputs[self.reward_col_name].type(dtype=torch.float).to(device)
        rewards = rewards.reshape(self.n_samples_per_prefix,-1)
        weights = nn.functional.softmax(self.inverse_beta * rewards, dim=0)
        return weights.unsqueeze(-1)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = self.compute_weights(inputs)
        out = model(inputs['input_ids'][:, :-1])
        b, s, v = out.logits.shape
        logits_reshaped = out.logits.reshape(-1,v) #einops.rearrange(out.logits, 'b s v -> (b s) v')
        targets_reshaped = inputs['input_ids'][:, 1:].reshape(-1) #einops.rearrange(inputs['input_ids'][:, 1:], 'b s -> (b s)')
        losses_reshaped = nn.functional.cross_entropy(logits_reshaped, targets_reshaped, reduce=False)
        losses = losses_reshaped.reshape(b,s) #einops.rearrange(losses_reshaped, '(b s) -> b s', b=b, s=s)
        losses = losses.reshape((self.n_samples_per_prefix, -1, s))
        weighted_losses = weights*losses
        weighted_losses = weighted_losses.sum(dim=0)
        return weighted_losses.mean()
    




#### General Idea
# We want to make the most of the huggingface Trainer class to implement invariant training for language models (see "invariang language modeling" paper)
# The caveat here is that our different environments are optimized on different trainers/learning objectives.

# What's my Idea to implement this ?
# I want to start off by having a single dataloader/dataset where one column of the dataset is the name of the trainer to use for the current sample. (we may want that this column is a list of trainers to use for the current sample if we should use more than 1)
# 1. Within the Trainer, Prepare the packing of the Dataloader so that in we only have samples of one trainer/environment at a time.
# 2. Use the training_step function to alternate optimizers. Tricky thing here that trainers use the _inner_training_loop function to train stuff on the optimizer (see here: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1919)
# Pseudo code of _inner_training_loop is more or less this:
    #a. Fancy stuff with state training + gpu stuff
    #b. Iterate over number of epochs:
        #b.1 Iterate over the dataloader:
            #b.1.1 training_step
            #b.1.2 if grad accumulation, do the accumulation or the optimizer step and zero the gradients
    #c. Final fancy stuff going on
# So basically problem here is that the optimizer step is not done in the training_step function but in _inner_training_loop. We want to do this without changing the HF library too much. Three ideas come to min rn:
    # Idea 1 : Optimizer step in the training_step function. Then have the optimizer of Invariant Training update nothing
    # Idea 2: switch optimizer and trainer in the training_step function. E.g., 
    #   trainer_to_use = inputs[self.trainer_name_col]
    #   self.optimizer = self.name_to_trainer[trainer_to_use].optimizer
    #   self.trainer = self.name_to_trainer[trainer_to_use]
    #  ...
    #This might come with some issues since trainers weren't designed to be switched in the middle of the training loop
    # Idea 3: change trainer/optimizer in a callback (e.g, on_step_end) (similar code to idea 2 but in a callback)

class InvariantModeling(Trainer):
    def __init__(self, name_to_trainer: Dict[str,Trainer], trainer_name_col ,*args, **kwargs):
        self.name_to_trainer = name_to_trainer  
        self.trainer_name_col = trainer_name_col
        super().__init__(*args, **kwargs)
    
    def set_trainer(self, name: str):
        self.trainer = self.name_to_trainer[name]
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.set_trainer(inputs[self.trainer_name_col])
        return self.trainer.training_step(model, inputs)