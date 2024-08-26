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

class BaseCtrlTokConfig(PretrainedConfig):
    model_type = "ctrl_tok"
    
    def __init__(
        self,
        control_token_to_id: Dict[str,int] = {},
        ctrl_token_head_temperature: float = 1.0,
        detach_ctrl_tok_clf = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert len(control_token_to_id) > 0, "control_token_to_id should be a list of at least one control token id"
        
        self.control_token_to_id = control_token_to_id
        self.num_control_tokens = len(control_token_to_id)
        self.detach_ctrl_tok_clf = detach_ctrl_tok_clf
        self.ctrl_token_head_temperature = ctrl_token_head_temperature
        
        
        
@dataclass
class SequenceClassifierOutputWithPastForCtrlTokens(ModelOutput):
    """
    Base class for outputs of control tokens models

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Either pause loss or language modeling loss (depending on the configuration of the model)
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
        control_token_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2)`):
            Pause logits.
        
    """

    loss: Optional[torch.FloatTensor] = None
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

        self.config.path_to_lm = self.language_model.config._name_or_path
        self.config.language_model_class = f"{self.language_model.__class__.__module__}.{self.language_model.__class__.__qualname__}"
        ############################################

        ########## Instantiating and resizing Control Token CLF and Embeddings ##########
        #get same dtype as model parameter
        torch_dtype = self.language_model.config.torch_dtype
        self.ctrl_tok_clf = nn.Linear(
            self.language_model.config.hidden_size,
            self.config.num_control_tokens + 1, # +1 because one of the tokens is "don't use a control token" control token,
            dtype=torch_dtype
        )
        self.control_token_to_id = config.control_token_to_id
        
        self._resize_input_embeds()
        self._resize_language_model_head()
        
        self.lm_head_ctrl_token_id = self.config.vocab_size #the last token is the control token for the language model head
        ############################################
        
        ## TODO: Can I remove the line here below ? Is this done automatically by the model ?
        # self.ctrl_tok_clf.to(self.language_model.device)
        
        #make generation config same as language model
        self.set_lm_generation_config()
        self.set_ctrl_token_temperature(config.ctrl_token_head_temperature)
        
    def _resize_input_embeds(self):
        """ Resize the input embeddings of the language model to account for the new control tokens """
        embeddings =  self.language_model.get_input_embeddings()
        self.language_model.set_input_embeddings(
            ExtendedEmbedding(
                embeddings,
                torch.nn.Embedding(
                    num_embeddings=self.config.num_control_tokens,
                    embedding_dim=self.language_model.config.hidden_size,
                    dtype=embeddings.weight.dtype,
                    device=embeddings.weight.device,
                ) 
            )
        )
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """ Personalized from_pretrained method to load the model from a pretrained model or a pretrained model folder
        
        :param args: Arguments to pass to the from_pretrained method (same as the from_pretrained method of the parent class)
        :param kwargs: Keyword arguments to pass to the from_pretrained method (same as the from_pretrained method of the parent class)
        """
        
        if pretrained_model_name_or_path is None:
            raise ValueError("pretrained_model_name_or_path should be provided")
        
        has_adapter_config = os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))
        
        ## Check If the model has a PeftConfig
        if pretrained_model_name_or_path is not None and has_adapter_config:
            peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            pause_clf_path = peft_config.base_model_name_or_path
        
            ctrl_tok_clf = cls.from_pretrained(pause_clf_path, *args, **kwargs)
          
            return PeftModel.from_pretrained(ctrl_tok_clf, pretrained_model_name_or_path, *args, **kwargs)
        # else load the model as usual
        else:
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    def save_pretrained(self, save_directory: str, *args, **kwargs):
        """ Save the model to a directory
        
        :param save_directory: The directory where to save the model
        :type save_directory: str
        """
        #get absolute path
        self.name_or_path = os.path.abspath(save_directory) 
        self.config.name_or_path = os.path.abspath(save_directory)
        super().save_pretrained(save_directory)
        
    
    def set_lm_generation_config(self):
        """ Set the generation config of the model to be the same as the language model """
        self.generation_config = self.language_model.generation_config
    
    def _validate_ctrl_token_id(self):
        """ Validate the control token id
        
        :param ctrl_token_id: The control token id to validate
        :type ctrl_token_id: int
        """
        for token, token_id in self.config.control_token_to_id.items():
            assert \
                token_id < self.config.vocab_size, \
                f"Control token id {token_id} of token {token} is out of range, the vocab size is {self.config.vocab_size}." + \
                f"You must set your control token ids as the last tokens of the vocabulary"
            og_vocab_size = self.get_original_vocab_size()
            assert \
                token_id >= og_vocab_size, \
                f"Control token id {token_id} of token {token} overlaps with the language model vocabulary." + \
                f"You must set your control token ids as the last tokens of the vocabulary (i.e. an id >= {og_vocab_size})"
            \
            
    def _resize_language_model_head(self):
        """ Resize the language model head to account for the new control tokens """
        if self.language_model.get_output_embeddings() is not None:
            if self.language_model.config.tie_word_embeddings:
                warnings.warn(f"The language model head is tied to the input embeddings " + \
                              f"but this is not compatible with the {self.__class__.__name__} model. " + \
                              f"Both the input embeddings and the output embeddings will be resized independently " + \
                              f"and the tie will be broken. Setting the tie_word_embeddings to False"
                )
                self.language_model.config.tie_word_embeddings = False
                           
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
            
        self.config.vocab_size = self.get_original_vocab_size() + self.config.num_control_tokens
        
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
            warnings.warn(
                f"Freezing the language model head, the name of the head to be frozen is {lm_head_name}, "+ \
                "make sure this is the correct head. If not, please provide the correct name as an argument."
            )
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
            warnings.warn(
                f"Freezing the language model head, the name of the head to be frozen is {lm_head_name}, "+ \
                "make sure this is the correct head. If not, please provide the correct name as an argument."
            )
            
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
            
            warnings.warn(
                f"Freezing the language model body, the name of the head to be frozen is {lm_head_name}, "+ \
                "make sure this is the correct head. If not, please provide the correct name as an argument."
            )
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
            
            warnings.warn(
                f"Freezing the language model body, the name of the head to be frozen is {lm_head_name}, "+ \
                "make sure this is the correct head. If not, please provide the correct name as an argument.")
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
                    
    def get_original_vocab_size(self):
        """ Get the original vocabulary size of the language model """
        return self.language_model.get_input_embeddings().original_embedding.weight.shape[0]
    
    def ctrl_tok_execute(self, labels: torch.LongTensor, token_name: str):
        """ Define the execution function for your control tokens here.
        
        A control token execution function should return:
            - CTRL_TOKEN_LABEL at positions where the control token.
            - LM_HEAD_LABEL at positions where the control token has not impacted the output (i.e., positions where its 0 are used to train the lm model and it's head)
            - IGNORE_LABEL at positions where the control token has impacted the output but should not be used to train the language model, the head or the control token head e.g., the pause token)
        
        constants come from src.utils.constants
        """
        raise NotImplementedError("Define the execution function for your control tokens here")
    
    def get_id_in_ctrl_tok_clf(self, token_id: Union[torch.Tensor,int]):
        """ Get the control token id in the control token classifier
        
        :param token_name: The name of the control token
        :type token_name: int
        """
 
        og_vocab_size = self.get_original_vocab_size()
        if isinstance(token_id, torch.Tensor):
            return torch.where(
                token_id < og_vocab_size,
                self.lm_head_ctrl_token_id - og_vocab_size,
                token_id - og_vocab_size,
            )
        else: 
            if token_id < og_vocab_size:
                return self.lm_head_ctrl_token_id - og_vocab_size 
            else:
                return token_id - og_vocab_size

 
    def get_ctrl_tok_labels(self, og_labels: torch.LongTensor):
        
        #compute exectute function for all control tokens and stack them
        stacked_ctrl_tok_labels = torch.stack(
            [self.ctrl_tok_execute(og_labels, token) for token in self.control_token_to_id.keys()],
            dim=0
        ) 
        
        #find the locations where a normal token is the label
        lm_head_token_locations = (stacked_ctrl_tok_labels == LM_HEAD_LABEL).all(dim=0)
        #find the locations where a control token is the label
        is_ctrl_tok_cnt = (stacked_ctrl_tok_labels == CTRL_TOKEN_LABEL).count_nonzero(dim=0)
        
        assert \
            (is_ctrl_tok_cnt <= 1).all(),\
            "There should be at most one control token per position. " + \
            "Your execution functions are probably not mutually exclusive"
        #locations where the label is a control token or a normal token
        label_occs = torch.logical_xor(is_ctrl_tok_cnt == 1, lm_head_token_locations)
        
        #is_ctrl_tok_cnt and lm_head_token_locations should not be both True at the same time
        assert \
            (label_occs == torch.logical_or(is_ctrl_tok_cnt, lm_head_token_locations)).all(), \
            "lm_head_token_locations and is_ctrl_tok_cnt should not be both True at the same time. "+ \
            "Make sure your execution functions are mutually exclusive"
        
        
        lm_head_labels = torch.where(label_occs, og_labels, IGNORE_LABEL)
        ctrl_tok_labels = self.get_id_in_ctrl_tok_clf(lm_head_labels)
        lm_head_labels = torch.where(
            ctrl_tok_labels == self.get_id_in_ctrl_tok_clf(self.lm_head_ctrl_token_id),
            lm_head_labels,
            IGNORE_LABEL
        )
        
        return lm_head_labels, ctrl_tok_labels
        
    def compute_loss(
        self,
        labels: torch.LongTensor,
        lm_logits: torch.FloatTensor,
        ctrl_tok_logits: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        assert labels is not None, "labels should not be None (labels are required for computing the loss)"
        
        #locations where the attention mask is 0 should be ignored (replace labels with IGNORE_LABEL)
        if attention_mask is not None:
            labels = torch.where(attention_mask == 0, IGNORE_LABEL, labels)
            
        #shift the labels and logits (For Decoder models, the first token should be ignored)
        shifted_labels = labels[..., 1:]
        shifted_lm_logits = lm_logits[..., :-1, :].contiguous()
        shifted_ctrl_tok_logits = ctrl_tok_logits[..., :-1, :].contiguous()
        
        #get the control token labels
        lm_head_labels, ctrl_tok_labels = self.get_ctrl_tok_labels(shifted_labels)

        #make sure both are contiguous
        lm_head_labels = lm_head_labels.contiguous()
        ctrl_tok_labels = ctrl_tok_labels.contiguous()
        
        #compute nll of control token head
        ctrl_tok_nlprobs = - torch.nn.functional.log_softmax(shifted_ctrl_tok_logits, dim=-1)
        #compute nll of lm head
        lm_nlprobs = - torch.nn.functional.log_softmax(shifted_lm_logits, dim=-1)
        
        #get the nll of the control token head and the lm head
        nll_ctrl_tok = ctrl_tok_nlprobs.gather(
            dim=-1,
            index= torch.clamp(ctrl_tok_labels, min=0).unsqueeze(-1)
        ).squeeze(-1)
        nll_lm = lm_nlprobs.gather(
            dim=-1,
            index=torch.clamp(lm_head_labels, min=0).unsqueeze(-1)
        ).squeeze(-1)
        
        #Note: it's normal that both use the lm_head_labels, 
        # because the control token labels are derived from the lm_head_labels
        
        #mask tokens that are ignored in loss
        mask_lm = (lm_head_labels == IGNORE_LABEL)
        nll_lm.masked_fill_(mask_lm, 0.0)
        
        mask_ctrl_tok = (ctrl_tok_labels == IGNORE_LABEL)
        nll_ctrl_tok.masked_fill_(mask_ctrl_tok, 0.0)    
        
        num_active_elements = mask_lm.numel() - (mask_lm & mask_ctrl_tok).count_nonzero()
        loss = (nll_lm + nll_ctrl_tok).sum()/num_active_elements
        return loss
    
    def forward_(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        
        if "return_dict" not in kwargs or kwargs["return_dict"] is None:
            kwargs["return_dict"] = True
            
        assert kwargs["return_dict"], "return_dict should be set to True"
        kwargs["output_hidden_states"] = True 
        
        labels = kwargs.pop("labels", None)
        
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *args,
            **kwargs
        )
        
        kwargs["labels"] = labels
                
        ctrl_tok_hidden_state = \
            outputs.hidden_states[-1].clone().detach() \
                if self.config.detach_ctrl_tok_clf else outputs.hidden_states[-1]
    
        ctrl_tok_logits = self.ctrl_tok_clf(ctrl_tok_hidden_state)
        
        lm_logits = outputs.logits
       
        #set logit of control tokens to -inf
        og_vocab_size = self.get_original_vocab_size()
        lm_logits[..., og_vocab_size:] = torch.finfo(lm_logits.dtype).min
        
        return lm_logits, ctrl_tok_logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions 
        
    def make_combined_logits(self, lm_logits: torch.FloatTensor, ctrl_tok_logits: torch.FloatTensor):
        """ Combine the logits of the language model and the control token classifier
        
        :param lm_logits: The logits of the language model
        :type lm_logits: torch.FloatTensor
        :param ctrl_tok_logits: The logits of the control token classifier
        :type ctrl_tok_logits: torch.FloatTensor
        :returns: logprobs of the combined logits
        """
        ctrl_tok_lprobs = torch.nn.functional.log_softmax(ctrl_tok_logits/self.ctrl_tokens_temperature, dim=-1)
        lm_lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1).clone()

        use_lm_head_id = self.get_id_in_ctrl_tok_clf(self.lm_head_ctrl_token_id)
        use_lm_head_lprobs = ctrl_tok_lprobs[..., use_lm_head_id].unsqueeze(-1)
        
        #use lm_head_id is actually always the last token of the control token classifier
        use_ctrl_tok_lprobs = ctrl_tok_lprobs[..., :-1] 

        og_vocab_size = self.get_original_vocab_size()
        #concatenate combined log prob vector
        lprobs = torch.cat([lm_lprobs[...,:og_vocab_size] + use_lm_head_lprobs, use_ctrl_tok_lprobs], dim=-1)
        
        return lprobs
        
    def forward(self,input_ids: torch.LongTensor = None , attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        
        lm_logits, ctrl_tok_logits, past_key_values, hidden_states, attentions  = \
            self.forward_(input_ids, attention_mask, *args, **kwargs)
        
        if kwargs.get("labels", None) is not None:
            loss = self.compute_loss(
                labels=kwargs["labels"],
                lm_logits=lm_logits,
                ctrl_tok_logits=ctrl_tok_logits,
            )
        else:
            loss = None
        
        logits = self.make_combined_logits(lm_logits, ctrl_tok_logits)
        
        return SequenceClassifierOutputWithPastForCtrlTokens(
            loss = loss,
            logits = logits,
            past_key_values = past_key_values,
            hidden_states = hidden_states,
            attentions = attentions,
            control_token_logits = ctrl_tok_logits,
            lm_logits = lm_logits,
        )
    
        
        
    
        

