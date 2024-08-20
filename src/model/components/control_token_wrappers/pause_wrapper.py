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
from src.model.components.control_token_wrappers.base_control_token_wrapper import \
    BaseCtrlTokConfig, BaseControlTokenWrapper
from src.utils.constants import CTRL_TOKEN_LABEL, LM_HEAD_LABEL, IGNORE_LABEL

class PauseCLFConfig(BaseCtrlTokConfig):    
    def __init__(
        self,
        pause_token_id: int = None,
        pause_token_name: str = "<|pause|>",
        **kwargs
    ):
        if kwargs.get("control_token_to_id") is not None:
            kwargs["control_token_to_id"][pause_token_name] = pause_token_id
        else:
            kwargs["control_token_to_id"] = {pause_token_name: pause_token_id}
            
        super().__init__(**kwargs)
        self.pause_token_id = pause_token_id
        self.pause_token_name = pause_token_name
        
class PauseClassifierWrapper(BaseControlTokenWrapper):
    config_class = PauseCLFConfig
    
    def ctrl_tok_execute(self, labels: torch.LongTensor, token_name: str, **kwargs):
        """ Execute function of pause token. Returns CTRL_TOKEN_LABEL anywhere the pause token is present in the labels tensor and LM_HEAD_LABEL elsewhere. 
        This function is used to determine whether each token in the input sequence is a control token (or part of a control token) or not. It's also used to determine the loss of the model.
        
        :param labels: torch.LongTensor of shape (batch_size, seq_len) containing the labels of the input sequence
        :param token_name: str, name of the token to execute
        :returns: torch.LongTensor of shape (batch_size, seq_len) containing the labels of the input sequence with the pause token replaced by CTRL_TOKEN_LABEL and the other tokens replaced by LM_HEAD_LABEL
        """
        if token_name == self.config.pause_token_name:
            return torch.where(labels == self.config.pause_token_id, CTRL_TOKEN_LABEL, LM_HEAD_LABEL)
        raise ValueError(f"Token name {token_name} not recognized")
        



def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tokenizers import AddedToken
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2")

    # Add pause token
    pause_token = AddedToken(
            "<|pause|>", 
            single_word=False, 
            lstrip=True, 
            rstrip=True
        )
    tokenizer.add_tokens([pause_token], special_tokens=True)
    #get the pause token id
    pause_token_id = tokenizer.convert_tokens_to_ids("<|pause|>")
    config = PauseCLFConfig(pause_token_id=pause_token_id, pause_token_name="<|pause|>")
    model = PauseClassifierWrapper(config = config, language_model = lm)
    return model, tokenizer

def test_inference():
    model, tokenizer = load_model()

    tokenized_seq = tokenizer("<|endoftext|> Hello <|pause|> world", return_tensors="pt")
    tokenized_seq["labels"] = tokenized_seq["input_ids"]
    
    output = model(**tokenized_seq)
    
    print("next_predicted_token", tokenizer.decode(output.logits[...,-1].argmax(dim=-1).item()))
    
def test_save_load_peft():
    def same_state_dicts(dict1, dict2):
        for key in dict1:
            if key not in dict2:
                return False
            elif isinstance(dict1[key], dict):
                if not same_state_dicts(dict1[key], dict2[key]):
                    return False
            elif not torch.allclose(dict1[key], dict2[key]):
                return False
        return True
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer
    model, tokenizer = load_model()
    model.save_pretrained("pre_lora")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        target_modules=["c_attn"]
    )
    
    model = get_peft_model(model, peft_config)
    model.save_pretrained("test_pause_model")
    tokenizer.save_pretrained("test_pause_model")
    
    model2 = PauseClassifierWrapper.from_pretrained("test_pause_model")
    tokenizer2 = AutoTokenizer.from_pretrained("test_pause_model")
    
    assert same_state_dicts(model.state_dict(), model2.state_dict())
    #delete directory
    os.system("rm -r test_pause_model")
    os.system("rm -r pre_lora")
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("testing inference...")
    test_inference()
    print("testing inference done")
    print("testing model loading and saving...")
    test_save_load_peft()
    print("testing model loading and saving done")
    
