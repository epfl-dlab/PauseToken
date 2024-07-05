import torch
import transformers
import pandas as pd
import numpy as np
import os
from collections import namedtuple
import torch
from tqdm import tqdm
import warnings
import json
import openai
import time
import os
import transformers
import pathlib
import argparse
from peft import AutoPeftModelForCausalLM
def generate(prompt: str, model, tokenizer):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt',truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=tokenized_prompt.input_ids,
                                attention_mask=tokenized_prompt.attention_mask,
                                do_sample=True,
                                max_new_tokens=256)

    output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    return output_decoded

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo-path', default=None)
    parser.add_argument('--test-data-path', default=None)
    parser.add_argument('--output-file-name', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    data = args.test_data_path
    dpo_model_dir = args.dpo_path
    output_file = args.output_file_name
      
    if dpo_model_dir:
        model = AutoPeftModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(dpo_model_dir)
    updated_data = []
    for i, pair in enumerate(json.load(open(data, 'r'))):
        #pair = json.loads(pair)
        new_data = {}
        prompt = pair['question']
        correct_answer = pair['answer']
        prompt = f"<s> [INST] ### Given the math word problem generate explanation and final answer. Question: {prompt} [/INST] \n### Answer:"
        generated = generate(prompt, model, tokenizer)
        new_data = pair.copy()
        new_data['output'] = generated
        
        updated_data.append(new_data) 
        with open(result+"/"+output_file, 'w') as json_file:
            json.dump(updated_data, json_file, indent=4, sort_keys=True)

        
if __name__ == "__main__":
    main()
