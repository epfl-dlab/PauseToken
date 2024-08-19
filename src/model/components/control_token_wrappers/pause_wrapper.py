from torch import nn
from transformers.utils import ModelOutput
from transformers import PreTrainedModel,AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from peft import PeftConfig, PeftModel
import warnings
import os
from src.model.components.embedding_wrappers import ExtendedEmbedding
PAUSE_CONFIG_NAME = "pause_clf.json"
SAFETENSORS_WEIGHTS_NAME = "pause_clf.safetensors"
WEIGHTS_NAME = "pause_clf.bin"
from transformers.modeling_utils import is_accelerate_available
if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module


class PauseCLFConfig(BaseCtrlTokConfig):
    model_type = "pause_clf"
    
    def __init__(
        self,
        pause_token_id: int = None,
        loss_type: str = "lm_loss",
        pause_loss_coeff: float = 0.5,
        lm_loss_coeff: float = 0.5,
        base_model_config: Optional[PretrainedConfig] = None,
        pause_temperature: float = 1.0,
        use_extended_embedding: bool = False,
        detach_pause_classifier: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pause_token_id = pause_token_id
        self.loss_type = loss_type
        self.pause_loss_coeff = pause_loss_coeff
        self.lm_loss_coeff = lm_loss_coeff
        self.base_model_config = base_model_config
        self.pause_temperature = pause_temperature
        self.use_extended_embedding = use_extended_embedding
        self.detach_pause_classifier = detach_pause_classifier
        
        
@dataclass
class SequenceClassifierOutputWithPastAndPauseLogits(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Either pause loss or language modeling loss (depending on the configuration of the model)
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for causal language models).
        pause_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Pause loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        pause_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2)`):
            Pause logits.
        
    """

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    pause_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    pause_logits: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    
    
class PauseClassifierWrapper(PreTrainedModel):
    config_class = PauseCLFConfig
    
    def __init__(self,config: PauseCLFConfig, language_model: PreTrainedModel = None):
        super().__init__(config)
        if language_model is not None:
            self.language_model = language_model
            self.config.path_to_lm = language_model.config._name_or_path
            # print("model name" , config.path_to_lm)
            # breakpoint()
            # self.config.base_model_config = language_model.config
            self.config.language_model_class = f"{language_model.__class__.__module__}.{language_model.__class__.__qualname__}"
        else:    
            self.language_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config.path_to_lm) #hydra.utils.instantiate({"_target_": config.base_model_class, "config": config.base_model_config})

        #get same dtype as model parameter
        torch_dtype = self.language_model.config.torch_dtype
        self.pause_classifier = nn.Linear(self.language_model.config.hidden_size, 2, dtype=torch_dtype)
        self.pause_token_id = config.pause_token_id
        
        if self.config.use_extended_embedding:
            embeddings =  self.language_model.get_input_embeddings()
            self.language_model.set_input_embeddings(
                ExtendedEmbedding(
                    embeddings,
                    torch.nn.Embedding(
                        num_embeddings=1,
                        embedding_dim=self.language_model.config.hidden_size,
                        dtype=embeddings.weight.dtype,
                        device=embeddings.weight.device
                    ) 
                )
            )
            self._resize_language_model_head()
            self.config.vocab_size = self.language_model.config.vocab_size + 1
        else:
            self._resize_if_necessary()
            self.config.vocab_size = self.language_model.config.vocab_size
        
        self.pause_classifier.to(self.language_model.device)
        
        assert config.loss_type in ["lm_loss", "pause_loss", "combined"], "loss_type should be either 'lm_loss', 'pause_loss' or 'combined'"
        self.loss_type = config.loss_type
        self.pause_loss_coeff = config.pause_loss_coeff
        self.lm_loss_coeff = config.lm_loss_coeff
        self.pause_temperature = config.pause_temperature if hasattr(config, "pause_temperature") else 1.0
        self.set_to_lm_logits()
        #make generation config same as language model
        self.generation_config = self.language_model.generation_config
    
    def set_lm_generation_config(self):
        self.generation_config = self.language_model.generation_config
    
    @classmethod
    def from_pretrained(cls,*args, **kwargs):
        # check if there is "adapter_config.json" in pretrained_model_name_or_path folder
        pretrained_model_name_or_path = None if len(args) == 0 else args[0]
        if pretrained_model_name_or_path is not None and os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json")):
            peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
            pause_clf_path = peft_config.base_model_name_or_path
            
            if len(args) > 1:
                pause_classifier = cls.from_pretrained(pause_clf_path, *args[1:], **kwargs)
            else:
                pause_classifier = cls.from_pretrained(pause_clf_path, **kwargs)
          
            return PeftModel.from_pretrained(pause_classifier, model_id = pretrained_model_name_or_path, **kwargs)
            
        else:
            return super().from_pretrained(*args, **kwargs)
    
    def _resize_language_model_head(self):
        
        if self.language_model.get_output_embeddings() is not None and not self.language_model.config.tie_word_embeddings:
            old_lm_head = self.language_model.get_output_embeddings()
            new_size = old_lm_head.weight.shape[0] + 1
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self.language_model._get_resized_embeddings(old_lm_head, new_size)
            else:
                new_lm_head = self.language_model._get_resized_lm_head(old_lm_head, new_size)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.language_model.set_output_embeddings(new_lm_head)

    def _resize_if_necessary(self):
        if self.pause_token_id == self.language_model.config.vocab_size:
            self.language_model.resize_token_embeddings(self.pause_token_id + 1)
        elif self.pause_token_id > self.language_model.config.vocab_size:
            raise ValueError("Model embedding size is smaller than pause token id. Please resize the model embedding size to fit the pause token id.")
        
    def set_pause_temperature(self, temparature: float):
        self.pause_temperature = temparature
       
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def freeze_pause_embeddings(self):
        for name, param in self.named_parameters():
            if "new_embedding" in name:
                param.requires_grad = False
    
    def unfreeze_pause_embeddings(self):
        for name, param in self.named_parameters():
            if "new_embedding" in name:
                param.requires_grad = True    
    
    def freeze_lm_embeddings(self):
        for name, param in self.named_parameters():
            if "original_embedding" in name:
                param.requires_grad = False
                
    def unfreeze_lm_embeddings(self):
        for name, param in self.named_parameters():
            if "original_embedding" in name:
                param.requires_grad = True
    
    def freeze_lm_head(self, lm_head_name: str = None):
        if lm_head_name is not None:
            lm_head = getattr(self.language_model, lm_head_name)
        else:
            lm_head_name,lm_head = [(name,param) for name,param in self.language_model.named_children()][-1]
            warnings.warn(f"Freezing the language model head, the name of the head to be frozen is {lm_head_name}, \
                make sure this is the correct head. If not, please provide the correct name as an argument.")
        for param in lm_head.parameters():
            param.requires_grad = False
            
    def unfreeze_lm_head(self,lm_head_name: str = None):
        if lm_head_name is not None:
            lm_head = getattr(self.language_model, lm_head_name)
        else:
            lm_head_name,lm_head = [(name,param) for name,param  in self.language_model.named_children()][-1]
            warnings.warn(f"Freezing the language model head, the name of the head to be frozen is {lm_head_name}, \
                make sure this is the correct head. If not, please provide the correct name as an argument.")
            
        for param in lm_head.parameters():
            param.requires_grad = True
            
    def freeze_lm_body(self, lm_head_name: str = None):
        
        if lm_head_name is not None:
            lm_head = getattr(self.language_model, lm_head_name)
        else:

            lm_head_name,lm_head  = [(name,param) for name,param in self.language_model.named_children()][-1]
            
            warnings.warn(f"Freezing the language model body, the name of the head to be frozen is {lm_head_name}, \
                make sure this is the correct head. If not, please provide the correct name as an argument.")
        #check if lm_head requires grad
        lm_head_requires_grad = any([p.requires_grad for p in lm_head.parameters()])
        self.freeze_language_model()
        if lm_head_requires_grad:
            self.unfreeze_lm_head(lm_head_name)
                
    def unfreeze_lm_body(self,lm_head_name: str = None):
        
        if lm_head_name is not None:
            lm_head = getattr(self.language_model, lm_head_name)
        else:

            lm_head_name,lm_head  = [(name,param) for name,param in self.language_model.named_children()][-1]
            
            warnings.warn(f"Freezing the language model body, the name of the head to be frozen is {lm_head_name}, \
                make sure this is the correct head. If not, please provide the correct name as an argument.")
        #check if lm_head requires grad
        lm_head_requires_grad = any([p.requires_grad for p in lm_head.parameters()])
        self.unfreeze_language_model()
        if not lm_head_requires_grad:
            self.freeze_lm_head(lm_head_name)
        
    def freeze_language_model(self):
        #Freeze everything except the pause classifier
        for param in self.language_model.parameters():
            param.requires_grad = False
            
    def unfreeze_language_model(self):
        #Unfreeze everything
        for param in self.language_model.parameters():
            param.requires_grad = True
    
    def freeze_pause_classifier(self):
        #Freeze the pause classifier (pause calssifiere in a nn.Linear)
        for param in self.pause_classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_pause_classifier(self):
        #Unfreeze the pause classifier (pause calssifiere in a nn.Linear)
        for param in self.pause_classifier.parameters():
            param.requires_grad = True
    
    def set_to_pause_logits(self):
        self.return_pause_logits_as_logits = True
        
    def set_to_lm_logits(self):
        self.return_pause_logits_as_logits = False
        
    def set_to_pause_loss(self):
        self.loss_type = "pause_loss"
    
    def set_to_lm_loss(self):
        self.loss_type = "lm_loss"
        
    def set_to_combined_loss(self):
        self.loss_type = "combined"
        
    
    def compute_loss(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        
        labels: Optional[torch.LongTensor] = None,
        pause_labels: Optional[torch.LongTensor] = None,
        lm_labels: Optional[torch.LongTensor] = None,
        ):
        
        #Make sure that either labels or (pause_labels and lm_labels) are provided
        labels_is_none = labels is None
        dual_labels_is_none = (pause_labels is None and lm_labels is None)
        assert (labels and not dual_labels_is_none) or (dual_labels_is_none and not labels_is_none), \
            f"Either labels or (pause_labels and lm_labels) should be provided, but got labels: {labels}, pause_labels: {pause_labels}, lm_labels: {lm_labels}"
        
        if not labels_is_none:
             # Create Pause labels (make sure it's on the same device as the model and a long tensor)
            pause_labels = torch.where(labels == self.pause_token_id, 1, 0).long().to(self.language_model.device)
            pause_labels = torch.where(labels == -100, -100, pause_labels).long().to(self.language_model.device)
            lm_labels = torch.where(labels == self.pause_token_id, -100, labels).long().to(self.language_model.device)
        
    def forward(self,input_ids: torch.LongTensor = None , attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        if "return_dict" not in kwargs or kwargs["return_dict"] is None:
            kwargs["return_dict"] = True
    
        assert kwargs["return_dict"], "return_dict should be set to True"
        #add output_hidden_states to the args
        kwargs["output_hidden_states"] = True  
        
        labels = kwargs.pop("labels",None)
        
        if labels is not None:
            # Create Pause labels (make sure it's on the same device as the model and a long tensor)
            pause_labels = torch.where(labels == self.pause_token_id, 1, 0).long().to(self.language_model.device)
            pause_labels = torch.where(labels == -100, -100, pause_labels).long().to(self.language_model.device)
            lm_head_labels = torch.where(labels == self.pause_token_id, -100, labels).long().to(self.language_model.device)
        
        else:
            pause_labels = None
        
        #add input_ids and attention_mask to the args
        outputs = self.language_model(input_ids= input_ids, attention_mask = attention_mask , *args, **kwargs)
        
        #get last hidden state from the outputs
        last_hidden_state = outputs.hidden_states[-1]
        
        if self.config.detach_pause_classifier:
            pause_hidden_state = last_hidden_state.clone().detach()
        else:
            pause_hidden_state = last_hidden_state
            
        pause_logits = self.pause_classifier(pause_hidden_state)
    
        ### INSERT PAUSE LOGIT IN LOGITS POSITION OF PAUSE TOKEN
        outputs.logits[..., self.pause_token_id] = torch.finfo(outputs.logits.dtype).min 
        
        lm_loss = None
        pause_loss = None
        combined_loss = None      
        if labels is not None and pause_labels is not None:
            if self.loss_type == "combined":
                shift_pause_labels = pause_labels[..., 1:].contiguous()
                shift_pause_logits = pause_logits[..., :-1, :].contiguous()
                
                shift_lm_labels = lm_head_labels[..., 1:].contiguous()
                shift_lm_logits = outputs.logits[..., :-1, :].contiguous()
                
                pause_log_probs = - torch.nn.functional.log_softmax(shift_pause_logits, dim=-1)
                lm_log_probs = - torch.nn.functional.log_softmax(shift_lm_logits, dim=-1)
         
                nll_pause = pause_log_probs.gather(dim=-1, index=torch.clamp(shift_pause_labels,min=0).unsqueeze(-1)).squeeze(-1)
                nll_lm = lm_log_probs.gather(dim=-1, index=torch.clamp(shift_lm_labels,min=0).unsqueeze(-1)).squeeze(-1)
                nll_lm.masked_fill_(shift_lm_labels.eq(-100), 0.0)
                nll_pause.masked_fill_(shift_pause_labels.eq(-100), 0.0)
    
                combined_loss = (nll_pause + nll_lm).mean()
                
            else:
                ### PAUSE LOSS
                # Shift so that tokens < n predict n
                shift_pause_logits = pause_logits[..., :-1, :].contiguous()
                shift_pause_labels = pause_labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_pause_logits = shift_pause_logits.view(-1, 2)
                shift_pause_labels = shift_pause_labels.view(-1)
                # Ensure tensors are on the same device
                shift_pause_labels = shift_pause_labels.to(shift_pause_logits.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                pause_loss = loss_fct(shift_pause_logits, shift_pause_labels)
                
                ## LM LOSS
                shift_lm_logits = outputs.logits[..., :-1, :].contiguous()
                shift_lm_labels = lm_head_labels[..., 1:].contiguous()
                # Flatten the tokens
                
                shift_lm_logits = shift_lm_logits.view(-1, self.config.vocab_size)
                shift_lm_labels = shift_lm_labels.view(-1)
                # Ensure tensors are on the same device
                shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                lm_loss = loss_fct(shift_lm_logits, shift_lm_labels)
        
            
        
        lm_logits = outputs.logits.clone()
        lm_loss = outputs.loss
        
        if not self.training:
            pause_prob = torch.nn.functional.softmax(pause_logits/self.pause_temperature, dim=-1)
            
            outputs.logits[..., -1 ,self.pause_token_id] = pause_prob[..., -1 ,1]
            # pause_prob = torch.nn.functional.softmax(pause_logits/self.pause_temperature, dim=-1)
            
            # sample_pause = torch.bernoulli(pause_prob[..., -1 ,1])
            
            # outputs.logits[..., self.pause_token_id] = float("-inf")
            
            # probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # probs[..., -1, :] = probs[..., -1, :] * (1-sample_pause).unsqueeze(-1)
            
            # probs[..., -1 ,self.pause_token_id] = sample_pause
            
            # outputs.logits = torch.log(probs)

            # outputs.logits = torch.where(
            #     torch.isnan(outputs.logits) | torch.isinf(outputs.logits),
            #     torch.finfo(outputs.logits.dtype).min,
            #     outputs.logits
            # )
        
        if labels is not None:
            if self.loss_type == "pause_loss":
                outputs.loss = pause_loss
            elif self.loss_type == "lm_loss":
                outputs.loss = lm_loss
            elif self.loss_type == "combined":
                outputs.loss = combined_loss

        if self.return_pause_logits_as_logits and self.training:
            outputs.logits = pause_logits
        
        return SequenceClassifierOutputWithPastAndPauseLogits(
            loss = outputs.loss,
            lm_loss= lm_loss,
            pause_loss = pause_loss,
            logits = outputs.logits,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
            pause_logits = pause_logits,
            lm_logits = lm_logits,
        )
    
        
        
    
        

