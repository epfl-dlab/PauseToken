from src.utils.constants import ANSWER_TEMPLATE, QUESTION_TEMPLATE
from datasets import Dataset
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedModel, LogitsProcessorList, StoppingCriteria
from transformers.generation.streamers import BaseStreamer
from typing import Optional, Callable, List, Dict, Any, Union
import torch
from tqdm import tqdm
import re
import numpy as np
import os
import json
from src.utils import (
    RankedLogger,
)
from trl import SFTTrainer

log = RankedLogger(__name__, rank_zero_only=True)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def inference_formatting_function(example, eos_token):
    inputs = []
    outputs = []
    for i in range(len(example["input"])):
        prompt = example["input"][i]
        input = f"{QUESTION_TEMPLATE}{prompt}\n\n{ANSWER_TEMPLATE}"
        output = f"{example['output'][i]}{eos_token}"
        inputs.append(input)
        outputs.append(output)
    return {"input": inputs, "output": outputs}

def reward_conditioning_inference_formatting_function(example, eos_token, correct_answer_feedback):
    inputs = []
    outputs = []
    for i in range(len(example["input"])):
        prompt = example["input"][i]
        input = f"{QUESTION_TEMPLATE}{prompt}\n\n{correct_answer_feedback}\n\n{ANSWER_TEMPLATE}"
        output = f"{example['output'][i]}{eos_token}"
        inputs.append(input)
        outputs.append(output)
    return {"input": inputs, "output": outputs}

def sft_formating_function(example, eos_token):
    data = []
    for i in range(len(example["input"])):
        prompt = example["input"][i]
        completion = example["output"][i]
        #add eos token to completion
        text = f"{QUESTION_TEMPLATE}{prompt}\n\n{ANSWER_TEMPLATE}{completion}{eos_token}"
        data.append(text)
    return data

def strip_pad_tokens(output: List[List[int]], pad_token_id: int) -> List[List[int]]:
    return [list(filter(lambda x: x != pad_token_id,seq))for seq in output]

def decode_and_strip_pad_tokens(output,pad_token_id, tokenizer):
    """ Decode and strip pad tokens from a list of sequences
    
    :param output: List of sequences
    :type output: List[List[int]]
    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :return: List of decoded sequences without pad tokens
    :rtype: List[str]
    """
    return tokenizer.batch_decode(strip_pad_tokens(output, pad_token_id))


def extract_answer(completion: str) -> str:
    """ Extracts the answer from the completion following the GSM8K dataset format
    
    :param completion: Completion
    :type completion: str
    :return: Extracted answer
    :rtype  str
    """
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def decode_and_strip_special_tokens(input_ids: Union[int, List[int], np.ndarray, torch.Tensor], tokenizer: PreTrainedTokenizer) -> str:
    """ Strip special tokens from a text
    
    :param input_ids: Input ids
    :type text: Union[int, List[int], np.ndarray, torch.Tensor]
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :return: Text without special tokens
    :rtype: str
    """
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def save_json(data: List[Dict[str, Any]], output_folder: str, file_name: str):
    """ Save a list of dictionaries to a json file
    
    :param data: Data to save
    :type data: List[Dict[str, Any]]
    :param path: Path to save the data
    :type path: str
    """
    path_to_output = os.path.join(output_folder, file_name)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #write results as json
    with open(path_to_output, 'w') as f:
        json.dump(data, f, indent=2)


def get_mean_and_std(values: List[Union[int, float]]) -> Dict[str, Union[float, int]]:
    """ Get the mean and standard deviation of a list of values
    
    :param values: List of values
    :type values: List[Union[int, float]]
    :return: Mean and standard deviation
    :rtype: Dict[str, Union[float, int]]
    """
    return {
        "mean": np.mean(values).item(),
        "std": np.std(values).item()
    }

def get_aggregated_metrics(data: List[Dict[str, Any]], metric_names) -> Dict[str, Union[float, int]]:
    aggregated_metrics = {}
    for name in metric_names:
        values = [d[name] for d in data]
        stat_dict = get_mean_and_std(values)
        aggregated_metrics[f"{name}_mean"] = stat_dict["mean"]
        aggregated_metrics[f"{name}_std"] = stat_dict["std"]
    return aggregated_metrics

def test_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    output_dir: Optional[str] = None,
    prompt_field: Optional[str] = "input",
    ground_truth_field: Optional[str] = "output",
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteria] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional[PreTrainedModel] = None,
    streamer: Optional[BaseStreamer] = None,
    negative_prompt_ids: Optional[torch.LongTensor] = None,
    generation_kwargs: Optional[Dict[str,Any]] = {},
    evaluation_metrics: Optional[Dict[str, Callable[[str, str], Union[int, float, bool]]]] = {},
    save_results: Optional[bool] = True,
    save_file_name: Optional[str] = "test_results.json",
    **kwargs
):
    """ Perform a "normal" rollout
    
    :param model: The model to use for generation
    :param tokenizer: The tokenizer to use for generation
    :param dataset: The dataset to generate from
    :param prompt_field: The field in the dataset that contains the prompts
    :param batch_size: The batch size to use for generation
    :param output_dir: The output directory to save the results
    :param generation_config: The generation config to use for generation
    :param logit_processor: The logit processor to use for generation
    :param stopping_criteria: The stopping criteria to use for generation
    :param prefix_allowed_tokens_fn: The prefix allowed tokens function to use for generation
    :param synced_gpus: Whether the gpus are synced
    :param assistant_model: The assistant model to use for generation
    :param streamer: The streamer to use for generation
    :param negative_prompt_ids: The negative prompt ids to use for generation
    :param generation_kwargs: The generation kwargs to use for generation
    :param evaluation_metrics: The evaluation metrics to evaluate the generated text
    """
    
    assert not generation_kwargs.get("return_dict_in_generate", False) \
        and not (model.config.return_dict_in_generate if hasattr(model.config, "return_dict_in_generate") else False), \
        "The `return_dict_in_generate` is not supported by test, please set it to False. If you want to use it consider making a PR to support it."
    
    og_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    was_in_training = model.training
    model.eval()
    
    res = []
    
    gt_is_in_dataset = ground_truth_field in dataset.column_names
    
    #iterate dataset in batchs
    for i in tqdm(range(0, len(dataset), batch_size),desc = "Rollout Step"): 
        if i+batch_size > len(dataset):
            batch = dataset[prompt_field][i:]
            ground_truths = dataset[ground_truth_field][i:] if gt_is_in_dataset else None
        else:
            batch = dataset[prompt_field][i:i + batch_size]
            ground_truths = dataset[ground_truth_field][i:i + batch_size] if gt_is_in_dataset else None
            
        tokenized_prompt = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        # get model dtype
        
        input_ids = tokenized_prompt.input_ids
        attention_mask = tokenized_prompt.attention_mask
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                logits_processor = logits_processor,
                stopping_criteria= stopping_criteria,
                prefix_allowed_tokens_fn= prefix_allowed_tokens_fn,
                synced_gpus= synced_gpus,
                assistant_model= assistant_model,
                streamer= streamer,
                negative_prompt_ids= negative_prompt_ids,
                **generation_kwargs
            )
            
        clean_text = decode_and_strip_pad_tokens(output, tokenizer.pad_token_id, tokenizer)
        tmp_res = \
            [
                {
                    "generated_text": text,
                    "tokenized_text": pred.tolist(),
                    "input": text.split(QUESTION_TEMPLATE)[1].split(ANSWER_TEMPLATE)[0].strip(),
                    "predicted_output": text.split(ANSWER_TEMPLATE)[1].strip(),
                    "ground_truth": ground_truths[i] if gt_is_in_dataset else None,
                }
                for i,(pred,text) in enumerate(zip(output, clean_text))
            ]
        
        for metric_name, metric_func in evaluation_metrics.items():
            for j in range(len(tmp_res)):
                gen_text_no_special_tokens = decode_and_strip_special_tokens(tmp_res[j]["tokenized_text"], tokenizer)
                tmp_res[j][metric_name] = metric_func(gen_text_no_special_tokens, ground_truths[j])
        res.extend(tmp_res)
    
    if save_results:
        log.info(f"Saving test results to {os.path.join(output_dir, save_file_name)}")
        save_json(res, output_dir, save_file_name)
    
    if was_in_training:
        model.train()
    tokenizer.padding_side = og_padding_side
    
    return get_aggregated_metrics(res, list(evaluation_metrics.keys()))

