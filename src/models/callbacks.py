from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

class SwitchLossCallback(TrainerCallback):
    def __init__(self):
        self.used_pause_loss = False
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.used_pause_loss:
            self.used_pause_loss = True
            kwargs["model"].set_to_pause_loss()
        else:
            self.used_pause_loss = False
            kwargs["model"].set_to_lm_loss()
