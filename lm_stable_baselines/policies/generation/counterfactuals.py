from transformers import PreTrainedModel, PreTrainedTokenizer
from torch import LongTensor
from typing import Dict, Any
from tqdm import tqdm
import torch
from lm_stable_baselines.utils import add_filler_tokens

def genererate_ctrltok_counterfactuals(
    language_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    predictions: LongTensor,
    generation_params: Dict[str, Any],
    ctrl_token_id: int,
    batch_size: int,
    filler_token_id: int,
    pad_length: int,
) -> LongTensor:
    
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    assert predictions.shape[0] == 1, "Use only one prediction at a time"
    
    og_padding_side = tokenizer.padding_side
    was_in_training = language_model.training
    language_model.eval()
    tokenizer.padding_side = "left"
    
    #find all positions where the ctrl token is present
    row, column = (predictions == ctrl_token_id).nonzero(as_tuple=True)
    assert len(row) == len(column), "Row and column should have the same length"
    
    counterfactuals = None
        
    counterfactual_inputs = []
    
    for i,j in zip(row, column):
        #copy the predictions tensor
        counterfactual_inputs.append(predictions[i,:j].clone())
        #replace the ctrl token with the filler token
    
    for i in tqdm(range(0, len(counterfactual_inputs), batch_size),desc = "Generating Counterfactuals"):
        if i+batch_size > len(counterfactual_inputs):
            batch = counterfactual_inputs[i:]
        else:
            batch = counterfactual_inputs[i:i + batch_size]
        
        #correctly pad the batch
        batch = tokenizer.pad({"input_ids": batch}, return_tensors="pt", padding=True)
        input_ids = batch["input_ids"].to(language_model.device)
        attention_mask = batch["attention_mask"].to(language_model.device)
        with torch.no_grad():
            #forward pass on the language model
            temperature = generation_params.generation_config.get("temperature", 1.0)
            outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            lm_logits = outputs.lm_logits[:,-1,:]/temperature
            #force the model to not predict the ctrl token
            lm_logits[..., ctrl_token_id] = torch.finfo(lm_logits.dtype).min
            probs = torch.nn.functional.softmax(lm_logits, dim=-1)
            #sample from the distribution
            sampled_tokens = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(sampled_tokens)], dim=-1)
            
            #generate rest of the couterfactual
            outputs = language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_params
            )
            
            padded_outputs = add_filler_tokens(outputs.cpu(), pad_length, filler_token_id)
            
            if counterfactuals is None:
                counterfactuals = padded_outputs
            else:
                counterfactuals = torch.cat([counterfactuals, padded_outputs], dim=0)
                
            
    if was_in_training:
        language_model.train()
    tokenizer.padding_side = og_padding_side
    
    return counterfactuals
    
    