import transformers
import pandas as pd
import numpy as np
import os
from collections import namedtuple
import torch
from tqdm import tqdm
import json
import os
import transformers
from datasets import load_dataset, Dataset
import argparse
from peft import AutoPeftModelForCausalLM
import hydra
from utils import dict_type, rollout,strip_special_tokens

# ANSWER_TEMPLATE = " ### Answer:"
# QUESTION_TEMPLATE = " ### Given the following math word problem question generate the correct final answer. Question: "
# REWARD_TEMPLATE = " ### Reward:"
# N_PAUSE_TOKENS_TEMPLATE = " ### Number of Pause Tokens:"
# N_PAUSE_TOKENS_AT_INFERENCE = 10

ANSWER_TEMPLATE = " ### Answer:"
QUESTION_TEMPLATE = " ### Given the following math word problem question generate the correct final answer. Question: "
REWARD_TEMPLATE = " ### Reward:"
N_PAUSE_TOKENS_TEMPLATE = " ### Number of Pause Tokens:"
N_PAUSE_TOKENS_AT_INFERENCE = 10

def formatting_original_dataset_func(example, task='gsm8k'):
    ## Task = argument or claim
    data = []
    for i in range(len(example['question'])):
        prompt = example['question'][i] #+ " Reasoning Chain: "+ example["predicted_rationale"][i]
        completion = example['answer'][i]
        reward = example['reward'][i]
        text = f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{REWARD_TEMPLATE} {reward}\n\n{N_PAUSE_TOKENS_TEMPLATE} {N_PAUSE_TOKENS_AT_INFERENCE}\n\n{ANSWER_TEMPLATE}"
        # text = f"<s> [INST]{QUESTION_TEMPLATE}{prompt} [/INST] \n\n{REWARD_TEMPLATE} {reward}\n\n{ANSWER_TEMPLATE}"
        data.append(text)     
    return {"text": data}

def add_max_reward(example,max_reward):
    return {"reward": [str(max_reward) for _ in range(len(example['question']))]}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--test-data-path', default=None)
    parser.add_argument('--reward', default={"_target_": "rewards.GSM8KDeltaAnswerReward"},type=dict_type)
    parser.add_argument('--output-folder',default = "../result")
    parser.add_argument("--output-filename", required=True)
    parser.add_argument("--run-first-n", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data = args.test_data_path
    model_dir = args.model_path
    reward = hydra.utils.instantiate(args.reward)
    maximal_reward = reward.get_max_reward()
    if model_dir:
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)

    results = []
    #load data (json with tranformers dataset)    
    dataset = load_dataset('json', data_files= data)["train"]
    if args.run_first_n is not None:
        tot_samps_to_runs = min(len(dataset), args.run_first_n)
        dataset = Dataset.from_dict(dataset[:tot_samps_to_runs])
    dataset = dataset.map(lambda x: add_max_reward(x,maximal_reward),batched=True)
    dataset= dataset.map(formatting_original_dataset_func, batched=True)
    results = rollout(model,tokenizer,dataset,prompt_field="text")
    
    json_res = []
    for question,result,gt,input_prompt in zip(dataset["question"],results,dataset["answer"],dataset["text"]):
        
        res = result["generated_text"].split(ANSWER_TEMPLATE)[-1]
            
        json_res.append(
            {
                "input": input_prompt,
                "answer": gt,
                "output": res,
                "question": question,
                "clean_output": strip_special_tokens(res,tokenizer=tokenizer),
            }
        )
        
    #Write results to file
    path_to_output = os.path.join(args.output_folder, args.output_filename)
    #make directory if it doesn't exist (use os makedirs)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    #write results as json
    with open(path_to_output, 'w') as f:
        json.dump(json_res, f, indent=4)
        
if __name__ == "__main__":
    main()