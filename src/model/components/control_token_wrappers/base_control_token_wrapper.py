from torch import nn
from transformers.utils import ModelOutput
from transformers import PreTrainedModel,PretrainedConfig,AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Callable, Dict
import torch
from peft import PeftConfig, PeftModel
import warnings
import os
from src.model.components.embedding_wrappers import ExtendedEmbedding
from transformers.modeling_utils import is_accelerate_available
if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module
from copy import deepcopy
from src.utils.constants import CTRL_TOKEN_LABEL, LM_HEAD_LABEL, IGNORE_LABEL


PAUSE_CONFIG_NAME = "pause_clf.json"
SAFETENSORS_WEIGHTS_NAME = "pause_clf.safetensors"
WEIGHTS_NAME = "pause_clf.bin"

class BaseCtrlTokConfig(PretrainedConfig):
    model_type = "pause_clf"
    
    def __init__(
        self,
        control_token_to_id: Dict[str,int],
        base_model_config: Optional[PretrainedConfig] = None,
        ctrl_token_head_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert len(control_token_to_id) > 0, "control_token_to_id should be a list of at least one control token id"
        assert "lm_head" not in control_token_to_id, "lm_head is a reserved token name"
        
        self.control_token_to_id = control_token_to_id
        self.num_control_tokens = len(control_token_to_id) + 1 # +1 because one of the tokens is "don't use a control token" control token
        
        self.base_model_config = base_model_config
        self.ctrl_token_head_temperature = ctrl_token_head_temperature
        
        
        
@dataclass
class SequenceClassifierOutputWithPastForCtrlTokens(ModelOutput):
    """
    Base class for outputs of control tokens models

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
    control_token_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    control_token_logits: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    
    
class BaseControlTokenWrapper(PreTrainedModel):
    config_class = BaseCtrlTokConfig
    
    def __init__(self,config: BaseCtrlTokConfig, language_model: PreTrainedModel = None):
        super().__init__(config)
        
        ########## Loading Language Model ##########
        #is the model is passed as an argument, use it, otherwise load the model from the path
        if language_model is not None:
            self.language_model = language_model
        else:    
            self.language_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config.path_to_lm)

        self.config.path_to_lm = language_model.config._name_or_path
        self.config.language_model_class = f"{language_model.__class__.__module__}.{language_model.__class__.__qualname__}"
        ############################################

        ########## Instantiating and resizing Control Token CLF and Embeddings ##########
        #get same dtype as model parameter
        torch_dtype = self.language_model.config.torch_dtype
        self.ctrl_tok_clf = nn.Linear(
            self.language_model.config.hidden_size,
            self.config.num_control_tokens,
            dtype=torch_dtype
        )
        self.control_token_to_id = config.control_token_to_id
        
        embeddings =  self.language_model.get_input_embeddings()
        self.language_model.set_input_embeddings(
            ExtendedEmbedding(
                embeddings,
                torch.nn.Embedding(
                    num_embeddings=self.config.num_control_tokens,
                    embedding_dim=self.language_model.config.hidden_size,
                    dtype=embeddings.weight.dtype,
                    device=embeddings.weight.device
                ) 
            )
        )
        self._resize_language_model_head()
        self.config.vocab_size = self.language_model.config.vocab_size + self.config.num_control_tokens
        ############################################
        
        ## TODO: Can I remove the line here below ? Is this done automatically by the model ?
        # self.ctrl_tok_clf.to(self.language_model.device)
        
        #make generation config same as language model
        self.set_lm_generation_config()
    
    @classmethod
    def from_pretrained(cls,*args, **kwargs):
        """ Personalized from_pretrained method to load the model from a pretrained model or a pretrained model folder
        
        :param args: Arguments to pass to the from_pretrained method (same as the from_pretrained method of the parent class)
        :param kwargs: Keyword arguments to pass to the from_pretrained method (same as the from_pretrained method of the parent class)
        """
        
        # check if there is "adapter_config.json" in pretrained_model_name_or_path folder
        pretrained_model_name_or_path = None if len(args) == 0 else args[0]
        
        has_adapter_config = os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))e
        
        ## Check If the model has a PeftConfig
        if pretrained_model_name_or_path is not None and has_adapter_config:
            peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
            pause_clf_path = peft_config.base_model_name_or_path
            
            if len(args) > 1:
                ctrl_tok_clf = cls.from_pretrained(pause_clf_path, *args[1:], **kwargs)
            else:
                ctrl_tok_clf = cls.from_pretrained(pause_clf_path, **kwargs)
          
            return PeftModel.from_pretrained(ctrl_tok_clf, model_id = pretrained_model_name_or_path, **kwargs)
        # else load the model as usual
        else:
            return super().from_pretrained(*args, **kwargs)
    
    
    def set_lm_generation_config(self):
        """ Set the generation config of the model to be the same as the language model """
        self.generation_config = self.language_model.generation_config
    
    def _validate_ctrl_token_id(self):
        """ Validate the control token id
        
        :param ctrl_token_id: The control token id to validate
        :type ctrl_token_id: int
        """
        for token, token_id in self.config.control_token_to_id.items():
            assert token_id < self.config.vocab_size, \
                f"Control token id {token_id} is out of range, the vocab size is {self.config.vocab_size}. You must set your control token ids as the last tokens of the vocabulary"
    
    def _resize_language_model_head(self):
        """ Resize the language model head to account for the new control tokens """
        if self.language_model.get_output_embeddings() is not None and not self.language_model.config.tie_word_embeddings:
            old_lm_head = self.language_model.get_output_embeddings()
            new_size = old_lm_head.weight.shape[0] + self.config.num_control_tokens
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
        
        self._validate_ctrl_token_id()

    def set_ctrl_token_temperature(self,temperature: float):
        """ Set the temperature of all control tokens to the same value
        
        :param temperature: The temperature to set to all control tokens
        :type temperature: float
        """
        self.ctrl_tokens_temperature = temperature
       
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """ Prepare the inputs for generation"""
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def freeze_ctrl_embeddings(self):
        """ Freeze the control token embeddings """
        for name, param in self.named_parameters():
            if "new_embedding" in name:
                param.requires_grad = False
    
    def unfreeze_ctrl_embeddings(self):
        """ Unfreeze the control token embeddings """
        for name, param in self.named_parameters():
            if "new_embedding" in name:
                param.requires_grad = True    
    
    def freeze_lm_embeddings(self):
        """ Freeze the language model embeddings """
        for name, param in self.named_parameters():
            if "original_embedding" in name:
                param.requires_grad = False
                
    def unfreeze_lm_embeddings(self):
        """ Unfreeze the language model embeddings """
        for name, param in self.named_parameters():
            if "original_embedding" in name:
                param.requires_grad = True
    
    def freeze_lm_head(self, lm_head_name: str = None):
        """ Freeze the language model head 
        
        :param lm_head_name: The name of the language model head to freeze. Default is None, which means the last head of the model will be frozen
        :type lm_head_name: str
        """
        if lm_head_name is not None:
            lm_head = getattr(self.language_model, lm_head_name)
        else:
            lm_head_name,lm_head = [(name,param) for name,param in self.language_model.named_children()][-1]
            warnings.warn(f"Freezing the language model head, the name of the head to be frozen is {lm_head_name}, \
                make sure this is the correct head. If not, please provide the correct name as an argument.")
        for param in lm_head.parameters():
            param.requires_grad = False
            
    def unfreeze_lm_head(self,lm_head_name: str = None):
        """ Unfreeze the language model head. Default is None, which means the last head of the model will be unfrozen
        
        :param lm_head_name: The name of the language model head to unfreeze. Default is None, which means the last head of the model will be unfrozen
        :type lm_head_name: str
        """
        if lm_head_name is not None:
            lm_head = getattr(self.language_model, lm_head_name)
        else:
            lm_head_name,lm_head = [(name,param) for name,param  in self.language_model.named_children()][-1]
            warnings.warn(f"Freezing the language model head, the name of the head to be frozen is {lm_head_name}, \
                make sure this is the correct head. If not, please provide the correct name as an argument.")
            
        for param in lm_head.parameters():
            param.requires_grad = True
            
    def freeze_lm_body(self, lm_head_name: str = None):
        """ Freeze the language model body. Default is None, which means the last head of the model will be frozen
        
        :param lm_head_name: The name of the language model head to keep unfrozen. Default is None, which means the last head of the model will remain unfrozen
        :type lm_head_name: str
        """
        
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
        """ Unfreeze the language model body. Default is None, which means the last head of the model will be unfrozen
        
        :param lm_head_name: The name of the language model head to keep frozen. Default is None, which means the last head of the model will remain frozen
        :type lm_head_name: str
        """
        
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
        """ Freeze the language model """
        #Freeze everything except the pause classifier
        for param in self.language_model.parameters():
            param.requires_grad = False
            
    def unfreeze_language_model(self):
        """ Unfreeze the language model """
        #Unfreeze everything
        for param in self.language_model.parameters():
            param.requires_grad = True
    
    def freeze_ctrl_tok_clf(self):
        """ Freeze the pause classifier """
        #Freeze the pause classifier (pause calssifiere in a nn.Linear)
        for param in self.ctrl_tok_clf.parameters():
            param.requires_grad = False
    
    def unfreeze_ctrl_tok_clf(self):
        """ Unfreeze the pause classifier """
        #Unfreeze the pause classifier (pause calssifiere in a nn.Linear)
        for param in self.ctrl_tok_clf.parameters():
            param.requires_grad = True
                    
    
    def ctrl_tok_execute(self, labels: torch.LongTensor, token_name: str):
        """ Define the execution function for your control tokens here.
        
        A control token execution function should return:
            - CTRL_TOKEN_LABEL at positions where the control token.
            - LM_HEAD_LABEL at positions where the control token has not impacted the output (i.e., positions where its 0 are used to train the lm model and it's head)
            - IGNORE_LABEL at positions where the control token has impacted the output but should not be used to train the language model, the head or the control token head e.g., the pause token)
        
        constants come from src.utils.constants
        """
        raise NotImplementedError("Define the execution function for your control tokens here")
    
    def exec_fn_to_label(self, exec_fn_res: torch.LongTensor, control_token_id: int):
        
        lm_head_token_id = self.control_token_to_id["lm_head"]
        
        label = exec_fn_res.clone()
        
        #mask for the control token id
        label[exec_fn_res == CTRL_TOKEN_LABEL] = control_token_id
        
        #mask for the lm head token id
        label[exec_fn_res == LM_HEAD_LABEL] = lm_head_token_id
                
        return label
    
    def make_lm_head_label(self, og_labels: torch.LongTensor, control_tokens_labels: List[torch.LongTensor]):
            
        lm_head_token_locations = \
            (
                torch.stack(
                    [control_tokens_labels[token] for token in control_tokens_labels],
                    dim=0) 
                == LM_HEAD_LABEL
            ).all(dim=0)
                 
        return torch.where(lm_head_token_locations, og_labels, IGNORE_LABEL)
    
    def get_ctrl_tok_labels(self, labels: torch.LongTensor):
        
        ctrl_tok_to_label = {}
        execute_fn_res = []
        
        for token, token_id in self.control_token_to_id.items():
            if token == "lm_head":
                continue
            exec_fn = self.ctrl_tok_execute(labels, token)
            ctrl_tok_to_label[token] = self.exec_fn_to_label(exec_fn, token_id)

        #get mask to create the lm_head label
        lm_head_labels = self.make_lm_head_label(labels, ctrl_tok_to_label)
                
        ctrl_tok_to_label["lm_head"] = lm_head_labels
        
        return ctrl_tok_to_label
        
    def compute_loss(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        logits: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        assert labels is not None, "labels should not be None (labels are required for computing the loss)"
        
        ctrl_tok_to_label = self.get_ctrl_tok_labels(labels)

        if logits is None:
            
            
    def get_all_logits(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        
        #make a copy of args and kwargs
        cp_kwargs = deepcopy(kwargs)
        cp_args = deepcopy(args)
        
        if "return_dict" not in cp_kwargs or cp_kwargs["return_dict"] is None:
            cp_kwargs["return_dict"] = True
            
        assert cp_kwargs["return_dict"], "return_dict should be set to True"
        cp_kwargs["output_hidden_states"] = True 
        
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *cp_args,
            **cp_kwargs
        )
                
        ctrl_tok_hidden_state = \
            outputs.hidden_states[-1].clone().detach() if self.config.detach_ctrl_tok_clf else  outputs.hidden_states[-1]
    
        
        ctrl_tok_logits = self.ctrl_tok_clf(ctrl_tok_hidden_state)
        
        lm_logits = outputs.logits
        
        
        
        
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
        
        if self.config.detach_ctrl_tok_clf:
            pause_hidden_state = last_hidden_state.clone().detach()
        else:
            pause_hidden_state = last_hidden_state
            
        pause_logits = self.ctrl_tok_clf(pause_hidden_state)
    
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
    
        
        
    
        

