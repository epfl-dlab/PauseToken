import torch
from torch import nn
from trl import SFTTrainer, DPOTrainer
import torch
import warnings
import numpy as np
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import Trainer, PreTrainedModel 
from typing import Dict, Union, Any, List, Optional
import importlib
from datasets import concatenate_datasets, Dataset
from typing import Callable, Dict,Tuple
# from samplers import BatchSubsetsSampler
from transformers import TrainingArguments, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.data.data_collator import DataCollator
from torch import nn
import torch
import copy
from transformers.utils import is_torch_xla_available
from src.model.components.control_token_wrappers import PauseClassifierWrapper
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

def pack_in_data_field(example):
    #check for fields of type torch.Tensor (use filter function)
    #TODO: This should be fixed to be more general
    string_fields = [key for key in example.keys() if isinstance(example[key], str) or (isinstance(example[key], list) and isinstance(example[key][0], str))]
    data_fields = [key for key in example.keys() if key not in string_fields]
    example["data"] = {key: example[key] for key in data_fields}     
    for col in data_fields:
        example.pop(col)
    return example

def add_train_method_name(example, col_name ,method_name):
    example[col_name] = method_name
    return example

def make_ds_compatible( datasets: List[Dataset]):
    def fill_missing_cols(example, cols, in_data_field = False):
        for col in cols:
            if in_data_field:
                example["data"][col] = None
            else:
                example[col] = None
        return example
    #make sure all datasets have the same columns in "data" field
    cols = [col for dataset in datasets for col in dataset[0]["data"].keys()]
    cols = set(cols)
    for i,dataset in enumerate(datasets):
        missing_cols = cols - set(dataset[0]["data"].keys())
        datasets[i] = dataset.map(lambda x: fill_missing_cols(x, missing_cols, in_data_field=True))
    #make sure all datasets have the same columns in the dataset
    cols = [col for dataset in datasets for col in dataset.column_names]
    cols = set(cols)
    for i,dataset in enumerate(datasets):
        missing_cols = cols - set(dataset.column_names)
        datasets[i] = dataset.map(lambda x: fill_missing_cols(x, missing_cols, in_data_field=False))
    return datasets




class SFTTrainerForPause(SFTTrainer):
    def __init__(self, pause_probability, *args, **kwargs):
        self.pause_probability = pause_probability
        self.losses = {"lm_loss": 0, "pause_loss": 0}
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
            
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        outputs = model(**inputs)
        
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        
        
        lm_loss = outputs["lm_loss"]
        
        uniform_pause_loss = self.uniform_pause_loss(outputs["control_token_logits"])
        
        loss = lm_loss + uniform_pause_loss
        
        self.losses["lm_loss"] += lm_loss.clone().detach().cpu()
        self.losses["pause_loss"] += uniform_pause_loss.clone().detach().cpu()

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
    
    def uniform_pause_loss(self, control_token_logits):
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        target = torch.zeros_like(control_token_logits)
        target[..., 0] = self.pause_probability
        target[..., 1] = 1 - self.pause_probability
        control_token_lprobs = nn.functional.log_softmax(control_token_logits, dim=-1)
        loss = kl_loss(control_token_lprobs, target)
        return loss
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            
            for loss_name, loss in self.losses.items():
                logs[loss_name] = round(loss.item() / (self.state.global_step - self._globalstep_last_logged), 4)
                self.losses[loss_name] -= self.losses[loss_name]

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            
            

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        

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
        if self.reward_col_name not in tokenized_dataset.column_names:
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

# class InvariantModelingTrainer(Trainer):
    
#     ## LEFT TO DO:
#     # - Include collator (creatted in collators.py)
#     # - Test it all out
#         # - check sampler
#         # - check collator
#         # - check trainer
#         # - check trainer switch
#         # - check prepare_dataset

#     def __init__(
#         self,
#         num_to_trainer_config: Dict[str,Dict],
#         name_to_formatting_func: Dict[str,Callable],
#         trainer_name_col: str,
#         columns_to_keep: List[str],
#         num_to_train_method: Dict[int,str] = None,
#         model: Union[PreTrainedModel, nn.Module] = None,
#         args: TrainingArguments = None,
#         data_collator: Optional[DataCollator] = None,
#         train_dataset: Optional[Dataset] = None,
#         eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
#         tokenizer: Optional[PreTrainedTokenizerBase] = None,
#         model_init: Optional[Callable[[], PreTrainedModel]] = None,
#         compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
#         callbacks: Optional[List[TrainerCallback]] = None,
#         optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
#         preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
#     ):
#         self.num_to_train_method = num_to_train_method
#         self.train_method_to_num = {v:k for k,v in num_to_train_method.items()}
#         self.trainer_name_col = trainer_name_col
#         self.name_to_formatting_func = name_to_formatting_func
#         self.columns_to_keep = columns_to_keep
#         train_dataset = train_dataset.select_columns(columns_to_keep)
#         if eval_dataset is not None:
#             eval_dataset = eval_dataset.select_columns(columns_to_keep)
#         trainers, train_dataset, eval_dataset,train_data_subset_samplers, eval_data_subset_samplers = \
#             self.prepare_dataset(
#                 train_dataset,
#                 eval_dataset,
#                 num_to_trainer_config,
#                 train_batch_size = args.train_batch_size,
#                 eval_batch_size = args.eval_batch_size,
#                 remove_unused_columns = args.remove_unused_columns
#             )
#         self.trainers = trainers
#         self.train_dataset = train_dataset
#         self.eval_dataset = eval_dataset
#         self.train_dataset_subset_samplers = train_data_subset_samplers
#         self.eval_dataset_subset_samplers = eval_data_subset_samplers
#         super().__init__(
#             model = model,
#             args = args,
#             data_collator = data_collator,
#             train_dataset = train_dataset,
#             eval_dataset = eval_dataset,
#             tokenizer = tokenizer,
#             model_init = model_init,
#             compute_metrics = compute_metrics,
#             callbacks = callbacks,
#             optimizers = optimizers,
#             preprocess_logits_for_metrics = preprocess_logits_for_metrics
#         )
#         self._signature_columns = ["data", self.trainer_name_col]
#         self.losses = {}
    
#     def create_optimizer_and_scheduler(self, num_training_steps: int):
#         super().create_optimizer_and_scheduler(num_training_steps)
#         # num_training_steps = num_training_steps // len(self.trainers)
#         # for trainer in self.trainers.values():
#         #     trainer.create_optimizer_and_scheduler(num_training_steps)
    
    
    
#     def prepare_dataset(self, train_dataset, eval_dataset, num_to_trainer_config,train_batch_size, eval_batch_size,remove_unused_columns):
#         #assert that the dataset has the column trainer_name_col, "rewards" and "text"
#         num_to_trainer_config = copy.deepcopy(num_to_trainer_config)
#         train_data_subset_samplers = {}
#         eval_data_subset_samplers = {}
#         train_data = []
#         eval_data = []
#         trainers = {}
#         last_idx_train = 0
#         last_idx_eval = 0

#         for name in num_to_trainer_config.keys():
#             #Fetch Train Data Related to the Trainer and format it
#             str_name = self.num_to_train_method[name]
            
#             trainer_train_data = train_dataset.filter(lambda x: str_name in x[self.trainer_name_col])
#             if self.name_to_formatting_func.get(str_name) is not None:
#                 trainer_train_data = trainer_train_data.map(self.name_to_formatting_func[str_name],batched=True, remove_columns=train_dataset.column_names)
#             #fetch trainer config
#             trainer_config = num_to_trainer_config[name]
#             trainer_config["train_dataset"] = trainer_train_data
#             if eval_dataset is not None:
#                 trainer_eval_data = eval_dataset.filter(lambda x: str_name in x[self.trainer_name_col])
#                 trainer_eval_data = trainer_eval_data.map(self.name_to_formatting_func[str_name], batched=True, remove_columns=eval_dataset.column_names)
#                 trainer_eval_data = trainer_eval_data.map(lambda x: add_train_method_name(x, col_name = self.trainer_name_col , method_name = name))
#                 trainer_config["eval_dataset"] = trainer_eval_data
#             # before_instantiation_cols = set(trainer_train_data.features)
#             cls = trainer_config.pop("trainer_class")
#             trainer = cls(**trainer_config)
#             train_data.append(
#                 trainer.train_dataset
#                 .map(lambda x: pack_in_data_field(x))
#                 .map(lambda x: add_train_method_name(x, col_name=self.trainer_name_col, method_name=name))
#             )
#             train_data_subset_samplers[name] = range(last_idx_train, last_idx_train + len(trainer.train_dataset))
#             last_idx_train += len(trainer.train_dataset)
#             trainer.train_dataset = None

#             if eval_dataset is not None:
#                 raise NotImplementedError("Not Tested yet on eval data")
#                 eval_data[name] = trainer.eval_dataset
#                 eval_data.append(trainer.eval_dataset)
#                 eval_data_subset_samplers[name] = range(last_idx_eval, last_idx_eval + len(trainer.eval_dataset))
#                 last_idx_eval += len(trainer.eval_dataset)
#                 trainer.eval_dataset = None
#             trainers[name] = trainer

#         train_data = make_ds_compatible(train_data)
#         train_data = concatenate_datasets(train_data)
#         train_data_subset_samplers = BatchSubsetsSampler(
#             subset_to_sampler = train_data_subset_samplers,
#             batch_size = train_batch_size,
#             resample_sample_till_all_done = True
#         )
        
#         if len(eval_data) > 0:
#             raise NotImplementedError("Not Tested yet on eval data")
#             eval_data = concatenate_datasets(eval_data)
#             eval_data_subset_samplers = BatchSubsetsSampler(
#                 subset_to_sampler = eval_data_subset_samplers,
#                 batch_size = eval_batch_size,
#                 resample_sample_till_all_done = False
#             )
#         else:
#             eval_data = None
#             eval_data_subset_samplers = None
#         #empty non usefule stuff on the GPU
#         torch.cuda.empty_cache()
#         return trainers, train_data, eval_data, train_data_subset_samplers, eval_data_subset_samplers
            
#     def set_trainer(self, name: str):
#         self.trainer = self.trainers[name]
#         # self.optimizer = self.trainer.optimizer
#         # self.lr_scheduler = self.trainer.lr_scheduler
        
#     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        
#         trainer_name = inputs.pop(self.trainer_name_col)
#         first_train_method = trainer_name[0].item()
#         self.current_train_method = self.num_to_train_method[first_train_method]
#         if not (first_train_method == trainer_name).all():
#             raise ValueError(f"All samples in the batch should have the same trainer,but got {inputs[self.trainer_name_col]}")
#         self.set_trainer(first_train_method)        
#         loss = self.trainer.training_step(model, inputs)
#         loss_name = f'{self.current_train_method}_loss'
#         if loss_name in self.losses:
#             self.losses[loss_name] += loss
#         else:
#             self.losses[loss_name] = loss
#         return loss
    
        
#     def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
#         if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
#             if is_torch_xla_available():
#                 xm.mark_step()

#             logs: Dict[str, float] = {}

#             # all_gather + mean() to get average loss over all processes
#             tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

#             # reset tr_loss to zero
#             tr_loss -= tr_loss

#             logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            
#             for ls in self.losses:
#                 ls_scalar =  self._nested_gather(self.losses[ls]).mean().item()
#                 logs[ls] = round(ls_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
#                 self.losses[ls] -= self.losses[ls]
            
#             if grad_norm is not None:
#                 logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
#             logs["learning_rate"] = self._get_learning_rate()

#             self._total_loss_scalar += tr_loss_scalar
#             self._globalstep_last_logged = self.state.global_step
#             self.store_flos()
#             self.log(logs)

#         metrics = None
#         if self.control.should_evaluate:
#             metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
#             self._report_to_hp_search(trial, self.state.global_step, metrics)

#             # Run delayed LR scheduler now that metrics are populated
#             if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                 metric_to_check = self.args.metric_for_best_model
#                 if not metric_to_check.startswith("eval_"):
#                     metric_to_check = f"eval_{metric_to_check}"
#                 self.lr_scheduler.step(metrics[metric_to_check])

#         if self.control.should_save:
#             self._save_checkpoint(model, trial, metrics=metrics)
#             self.control = self.callback_handler.on_save(self.args, self.state, self.control)

#     def _get_train_sampler(self):
#         return self.train_dataset_subset_samplers
    
#     def _get_eval_sampler(self):
#         return self.eval_dataset_subset_samplers
        
#     def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
#         raise NotImplementedError("Not implemented yet")

# class PauseHeadDPO(DPOTrainer):
    
#     def __init__(self, pause_token_id ,*args, **kwargs):
#         self.pause_token_id = pause_token_id
#         super().__init__(*args, **kwargs)
#         if self.ref_model is not None:
#             self.ref_model.set_to_pause_logits()

    
#     def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
#         batch = super().tokenize_row(feature, model)
#         for key in batch.keys():
#             if "_labels" in key:
#                 batch[key] = list(
#                     map(
#                         lambda x: int(x == self.pause_token_id) if x != self.label_pad_token_id else self.label_pad_token_id,
#                         batch[key]))
#         return batch        
#     def compute_loss(self, model, inputs, return_outputs=False):
#         self.model.set_to_pause_logits()
#         model.set_to_pause_logits()
#         return super().compute_loss(model, inputs, return_outputs)

# class WeightedSFTTrainerLMLoss(WeightedSFTTrainer):
#     def __init__(self, pause_token_id ,*args, **kwargs):
#         self.pause_token_id = pause_token_id
#         super().__init__(*args, **kwargs)
        
#     def compute_loss(self, model, inputs, return_outputs=False):
#         self.model.set_to_lm_logits()
#         model.set_to_lm_logits()
#         weights = self.compute_weights(inputs)
#         out = model(inputs['input_ids'][:, :-1])
#         b, s, v = out.logits.shape
#         logits_reshaped = out.logits.reshape(-1,v) #einops.rearrange(out.logits, 'b s v -> (b s) v')
#         targets_reshaped = inputs['input_ids'][:, 1:].reshape(-1) #einops.rearrange(inputs['input_ids'][:, 1:], 'b s -> (b s)')
#         targets_reshaped = torch.where(targets_reshaped == self.pause_token_id, -100, targets_reshaped)
#         losses_reshaped = nn.functional.cross_entropy(logits_reshaped, targets_reshaped, reduce=False, ignore_index=-100)
#         losses = losses_reshaped.reshape(b,s) #einops.rearrange(losses_reshaped, '(b s) -> b s', b=b, s=s)
#         losses = losses.reshape((self.n_samples_per_prefix, -1, s))
#         weighted_losses = weights*losses
#         weighted_losses = weighted_losses.sum(dim=0)
#         return weighted_losses.mean()
    
# class SFTTrainerLMLoss(SFTTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         self.model.set_to_lm_logits()
#         self.model.set_to_lm_loss()
#         model.set_to_lm_logits()
#         model.set_to_lm_loss()
#         return super().compute_loss(model, inputs, return_outputs)
    
# class SFTTrainerPauseLoss(SFTTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         self.model.set_to_pause_logits()
#         self.model.set_to_pause_loss()
#         model.set_to_pause_logits()
#         model.set_to_pause_loss()
#         return super().compute_loss(model, inputs, return_outputs)