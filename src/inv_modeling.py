from transformers import Trainer
from typing import Dict, Union, Any
import torch
from torch import nn


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
    # Idea1 : Optimizer step in the training_step function. Then have the optimizer of Invariant Training update nothing
    # Idea2: switch optimizer and trainer in the training_step function. E.g., 
    #   trainer_to_use = inputs[self.trainer_name_col]
    #   self.optimizer = self.name_to_trainer[trainer_to_use].optimizer
    #   self.trainer = self.name_to_trainer[trainer_to_use]
    #  ...
    #This might come with some issues since trainers weren't designed to be switched in the middle of the training loop
    # Idea3: change trainer/optimizer in a callback (e.g, on_step_end) (similar code to idea 2 but in a callback)

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
        
