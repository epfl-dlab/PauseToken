import torch
import os
import json 
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, GenerationConfig,LogitsProcessorList
from transformers.pipelines.pt_utils import KeyDataset
from constraints import PauseGroundTruthConstraint,PauseLogitsProcessor
import re
import pathlib
import time
from tqdm import tqdm
import argparse

def count_num_token_occurences(token_id, tokenizer, text):
    """ Count the number of occurences of a token in a text
    
    :param token_id: Token id to count
    :type token_id: int
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param text: Text or list of texts to count the token occurences
    :type text: Union[str, List[str]]
    :return: Number of occurences of the token in the text
    :rtype: Union[int, List[int]]
    """  
    tokenized_text = tokenizer(text)["input_ids"]
    if isinstance(text, str):
        return len(list(filter(lambda x: x == token_id, tokenized_text)))
    elif isinstance(text, list):
        return list(
            map(
                lambda x: len(list(filter(lambda y: y == token_id, x))),
                tokenized_text
            )
        )   
    else:
        raise ValueError("Text must be either a string or a list of strings.")
        
def strip_special_tokens(text,tokenizer):
    """ Strip special tokens from a text
    
    :param text: Text to strip special tokens from
    :type text: str
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :return: Text without special tokens
    :rtype: str
    """
    tokenized_text = tokenizer(text)["input_ids"]
    return tokenizer.decode(tokenized_text, skip_special_tokens=True)

def strip_pause_tokens(text,tokenizer,pause_token_id):
    """ Strip pause tokens from a text
    
    :param text: Text to strip pause tokens from
    :type text: str
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param pause_token_id: Pause token id
    :type pause_token_id: int
    :return: Text without pause tokens
    :rtype: str
    """
    pause_token = tokenizer.decode(pause_token_id)
    return text.replace(pause_token,"")

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
    return tokenizer.batch_decode([list(filter(lambda x: x != pad_token_id,seq))for seq in output])
        
def pause_ground_truth_constrained_rollout(
    model,
    tokenizer,
    dataset: Dataset,
    prompt_field,
    ground_truth_field,
    pause_token_id,
    generation_args= {
        "temperature": 0.9,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "max_length": 400,
    },
    batch_size = 8,
    n_samps_per_prompt = 1,
    include_gt = False,
    pause_temperature = 1.0
):
    """ Perform a rollout with pause ground truth constraint. Meaning that at generation, the model can either generate as next token the pause token or the next ground truth token.
    
    :param model: Model
    :type model: transformers.PreTrainedModel
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param dataset: Dataset
    :type dataset: datasets.Dataset
    :param prompt_field: Prompt field name. The field in the dataset that contains the prompt to feed to the model
    :type prompt_field: str
    :param ground_truth_field: Ground truth field name. The field in the dataset that contains the ground truth completion
    :type ground_truth_field: str
    :param pause_token_id: Pause token id
    :type pause_token_id: int
    :param generation_args: Generation arguments (Huggingface Transformers generation arguments)
    :type generation_args: dict
    :param batch_size: Batch size
    :type batch_size: int
    :param n_samps_per_prompt: Number of samples to generate per prompt
    :type n_samps_per_prompt: int
    :param include_gt: Include the ground truth (w/out pauses) in the generated samples
    :type include_gt: bool
    :param pause_temperature: Pause temperature (see 5.3 of Overleaf Amortized Search For Language Model Decoding)
    :type pause_temperature: float
    :return: List of generated samples
    :rtype: List[dict]
    """
    constraint_module = PauseGroundTruthConstraint(tokens_to_filter=[pause_token_id,tokenizer.pad_token_id], max_tokens=generation_args.get("max_length"))
    pause_temperature_logit_processor = PauseLogitsProcessor(pause_token_id=pause_token_id, pause_temperature= pause_temperature, softmax_temperature=generation_args.get("temperature",1.0))
    was_in_training = model.training
    og_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    res = []
    

    #iterate dataset in batchs
    for i in tqdm(range(0, len(dataset), batch_size),desc = "Rollout Step"): 
        
        if i+batch_size > len(dataset):
            batch = dataset[prompt_field][i:]
            ground_truths = list(map(lambda x: strip_pause_tokens(x, tokenizer, pause_token_id), dataset[ground_truth_field][i:]))
            
        else:
            batch = dataset[prompt_field][i:i + batch_size]
            ground_truths = list(map(lambda x: strip_pause_tokens(x, tokenizer, pause_token_id), dataset[ground_truth_field][i:i + batch_size]))
            
        bs = len(batch)
        tokenized_prompt = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
        tokenized_ground_truth = tokenizer(ground_truths, padding=False)["input_ids"]
        prefix_allowed_tokens_fn = constraint_module.get_prefix_allowed_tokens_fn(
            batch_info={"tokenized_ground_truths": tokenized_ground_truth, "pause_token_id": pause_token_id, "pad_token_id": tokenizer.pad_token_id}
        )
        tmp_res_per_batch_id = {j: [] for j in range(bs)}
        for _ in range(n_samps_per_prompt):
            
            with torch.no_grad():
                output = model.generate(
                    input_ids=tokenized_prompt.input_ids,
                    attention_mask=tokenized_prompt.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    logits_processor=LogitsProcessorList([pause_temperature_logit_processor]),
                    **generation_args
                )
            
            for j,text in enumerate(decode_and_strip_pad_tokens(output,tokenizer.pad_token_id,tokenizer)):
                tmp_res_per_batch_id[j].append({"generated_text": text, prompt_field: batch[j]}) 

        if include_gt:
            for i,gt in enumerate(tokenizer.batch_decode(tokenized_ground_truth)):
                tmp_res_per_batch_id[i].append({"generated_text": gt, prompt_field: batch[j]})
            

        for batch_id, texts in tmp_res_per_batch_id.items():
            res.extend(texts)      
    if was_in_training:
        model.train()
    tokenizer.padding_side = og_padding_side
    return res


def rollout(
    model,
    tokenizer,
    dataset: Dataset,
    prompt_field ,
    generation_args= {
        "temperature": 0.9,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "max_new_tokens": 400,
    },
    batch_size=8
    ):
    """ Perform a "normal" rollout
    
    :param model: Model
    :type model: transformers.PreTrainedModel
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param dataset: Dataset
    :type dataset: datasets.Dataset
    :param prompt_field: Prompt field name. The field in the dataset that contains the prompt to feed to the model
    :type prompt_field: str
    :param generation_args: Generation arguments (Huggingface Transformers generation arguments)
    :type generation_args: dict
    :param batch_size: Batch size
    :type batch_size: int
    """
    og_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    was_in_training = model.training
    model.eval()
    res = []
    #iterate dataset in batchs
    for i in tqdm(range(0, len(dataset), batch_size),desc = "Rollout Step"): 
        if i+batch_size > len(dataset):
            batch = dataset[prompt_field][i:]
        else:
            batch = dataset[prompt_field][i:i + batch_size]
            
        tokenized_prompt = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids=tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                **generation_args
            )
            
        res.extend(decode_and_strip_pad_tokens(output,tokenizer.pad_token_id, tokenizer))
        
    res = [{"generated_text": text} for text in res]
    if was_in_training:
        model.train()
    tokenizer.padding_side = og_padding_side
    return res
        
    
def dict_type(string):
    """ Convert a string to a dictionary
    
    :param string: A string that represents a dictionary
    :type string: str
    :return: A dictionary
    :rtype: dict
    """
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid dictionary format. Must be a valid JSON string.")
    

def save_to(data, name, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False)
    
    
def low_resource_data(data):
    ### data format = ['prompt': .., 'chosen': .., 'rejected': ..}
    new_data = []
    chosen_for_now = []
    for sample in data:
        if chosen_for_now.count(sample['chosen']) > 2:
            continue
        
        new_data.append(sample)
        chosen_for_now.append(sample['chosen'])
    return new_data

def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, model_args=None):
    import openai
    if model_args is None:
        model_args = {}
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, **model_args)
            if return_text:
                return response["choices"][0]["message"]["content"].strip()
            return response
        except Exception as e:
            print(e)
            print("Timed out. Waiting for 1 minute.")
            time.sleep(60)
            continue

def get_gpt_response(input_, model='gpt-3.5-turbo', necessary_tokens=None):
    return chat_completion([{"role": "assistant", "content": input_}], model=model, return_text=True, model_args={
                    "temperature": 0.4,
                    "max_tokens": 150 if not necessary_tokens else necessary_tokens,
                    "top_p": 0.4,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                    })
        
        
def process_gpt_output(output_text):
    try:
        result_dict = json.loads(output_text)
        return result_dict
    except json.JSONDecodeError:
        start_index = output_text.find('{')
        end_index = output_text.rfind('}')
        if start_index != -1 and end_index != -1:
            extracted_dict_text = output_text[start_index:end_index + 1]
            try:
                result_dict = json.loads(extracted_dict_text)
                return result_dict
            except json.JSONDecodeError:
                print("Failed to extract a valid dictionary from the text.")
                return None
        else:
            print("No dictionary found in the text.")
            return None
        
def remove_incomplete_last_sentence(text):
    import nltk
    sentences = nltk.sent_tokenize(text)

    last_sentence = sentences[-1]
    if last_sentence.endswith(('?', '.', '!')):
        return text
    else:
        return ' '.join(sentences[:-1])

def sanitize(sentence):
    sanitized_sentence = re.sub(r'[^\w\s.,!?\'-]', '', sentence)
    sanitized_sentence = re.sub(r'\s+', ' ', sanitized_sentence).strip()
    return sanitized_sentence

def get_data(data_dir, split='train', return_type='dataset', with_equivalence=False):
    if data_dir[-1] != '/': 
        data_dir += '/'
    with open(f'{data_dir+split}.json', 'r') as f:
        df = json.load(f)

    df = pd.DataFrame(df)
    if with_equivalence: 
        with open(data_dir + f'{split}_equivalent.json', 'r') as f:
            equivalence = json.load(f)
        
        df2 = df.copy()    
        df2.drop_duplicates(subset=['chosen'], inplace=True)
        
        d = {'prompt': df2.prompt, 'chosen': df2.chosen, 'rejected': [equivalent['argument'] for equivalent in equivalence]}
        
        new_df = pd.DataFrame(data=d).reset_index(drop=True)
        df = pd.concat([df, new_df])
 
        df = df.reset_index(drop=True)

    if return_type == 'df':
        return df
    
    for i, item in df.iterrows():
        prompt = df.iloc[i]['prompt']
        if 'supporting argument' in prompt:
            prompt = prompt.replace('supporting argument', 'SUPPORTING argument of about 20 words')
        else:
            prompt = prompt.replace('counter argument', 'COUNTER argument of about 20 words')

        prompt += '\n### Argument:'
        df.iloc[i]['prompt'] = prompt
    
    return Dataset.from_dict(df)

def get_training_args(args):
    return TrainingArguments(
            output_dir=args.output_dir,               
            overwrite_output_dir=False,                  
            num_train_epochs=args.n_epochs,                   
            per_device_train_batch_size=args.batch_size,         
            learning_rate=args.learning_rate,                      
            warmup_steps=args.warmup_steps,                           
            weight_decay=args.weight_decay,                         
            adam_epsilon=args.adam_epsilon,                         
            save_steps=args.save_steps,                       
            logging_steps=args.logging_steps,                      
            save_total_limit=2,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )       