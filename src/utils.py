import torch
import os
import json 
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, GenerationConfig
from transformers.pipelines.pt_utils import KeyDataset

import re
import pathlib
import time
from tqdm import tqdm
import argparse

def count_num_token_occurences(token_id, tokenizer, text):    
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
    tokenized_text = tokenizer(text)["input_ids"]
    return tokenizer.decode(tokenized_text, skip_special_tokens=True)

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
    
    def decode_and_strip_pad_tokens(output,pad_token_id):
        decoded_seq = tokenizer.batch_decode(output)
        return tokenizer.batch_decode(
            [list(filter(lambda x: x != pad_token_id,seq)) for seq in tokenizer(decoded_seq)["input_ids"]]
        )
    
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
            
        res.extend(decode_and_strip_pad_tokens(output,tokenizer.pad_token_id))
        
    # pipe = pipeline(
    #     task = "text-generation",
    #     tokenizer = tokenizer,
    #     model = model,
    #     torch_dtype = "auto",
    #     device_map="auto",
    #     **generation_args
    # )
    
    # res = pipe(KeyDataset(dataset, prompt_field))
    
    # breakpoint()
    # res = list(
    #     map(
    #         lambda x: {
    #             "generated_text": x[0]["generated_text"],
    #         },
    #         res
    #     )
    # )
    res = [{"generated_text": text} for text in res]
    if was_in_training:
        model.train()
        
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