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
from pause_classifier_wrapper import PauseClassifierWrapper

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

SUPPORTED_INF_METHOD = ["wsft", "rc","rc_w_n_pause" ,"dpo","prepause"]
PAUSE_SYMBOL = "<|pause|>"

def formatting_original_dataset_func(example, train_method='wsft'):
    def formatting_original_dataset_func_wsft(example):
        data = []
        for i in range(len(example['question'])):
            prompt = example['question'][i]
            data.append(f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{ANSWER_TEMPLATE}")
        return {"text": data}
    
    def formatting_original_dataset_func_prepause(example):
        data = []
        ten_pauses = "".join([PAUSE_SYMBOL for _ in range(N_PAUSE_TOKENS_AT_INFERENCE)])
        for i in range(len(example['question'])):
            prompt = example['question'][i]
            data.append(f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{ANSWER_TEMPLATE}{ten_pauses}")
        return {"text": data}
    
    
    def formatting_original_dataset_func_rc(example, with_pause):
        data = []
        for i in range(len(example['question'])):
            prompt = example['question'][i] #+ " Reasoning Chain: "+ example["predicted_rationale"][i]
            reward = example['reward'][i]
            if with_pause:
                text = f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{REWARD_TEMPLATE} {reward}\n\n{N_PAUSE_TOKENS_TEMPLATE} {N_PAUSE_TOKENS_AT_INFERENCE}\n\n{ANSWER_TEMPLATE}"
            else:
                text = f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{REWARD_TEMPLATE} {reward}\n\n{ANSWER_TEMPLATE}"
            data.append(text) 
        return {"text": data} 
    
    if train_method in ['wsft',"dpo"]:
        return formatting_original_dataset_func_wsft(example)
    elif train_method == 'rc':
        return formatting_original_dataset_func_rc(example,with_pause = False)
    elif train_method == 'rc_w_n_pause':
        return formatting_original_dataset_func_rc(example, with_pause = True)
    elif train_method == 'prepause':
        return formatting_original_dataset_func_prepause(example)
    else:
        raise ValueError(f"train_method {train_method} not supported. Supported methods are {SUPPORTED_INF_METHOD}")    


def add_max_reward(example,max_reward):
    return {"reward": [str(max_reward) for _ in range(len(example['question']))]}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--test-data-path', default=None)
    parser.add_argument('--reward', default={"_target_": "rewards.GSM8KDeltaAnswerReward"},type=dict_type)
    parser.add_argument('--output-folder',default = "../result")
    parser.add_argument("--output-filename", required=True)
    parser.add_argument("--train-method", default="wsft", type=str)
    parser.add_argument("--run-first-n", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    
    assert args.train_method in SUPPORTED_INF_METHOD, f"train method {args.train_method} not supported. Supported methods are {SUPPORTED_INF_METHOD}"
    
    data = args.test_data_path
    
    model_dir = args.model_path
    
    reward = hydra.utils.instantiate(args.reward)
    
    maximal_reward = reward.get_max_reward()
    
    if model_dir:
    
        # try:
        # model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map='auto')
        # tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    
        # except:
        #     try:
        # model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.bfloat16)
        # tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        #     except:
                #put all on gpu (without using auto)
        model = PauseClassifierWrapper.from_pretrained(model_dir,device_map= "cuda:0" ,torch_dtype= torch.bfloat16)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        
    results = []
    tokenizer.pad_token=tokenizer.unk_token
    #load data (json with tranformers dataset)    
    
    dataset = load_dataset('json', data_files= data)["train"]
    
    if args.run_first_n is not None:
        tot_samps_to_runs = min(len(dataset), args.run_first_n)
        dataset = Dataset.from_dict(dataset[:tot_samps_to_runs])
    
    if args.train_method in ["rc_w_n_pause", "rc"]:
        dataset = dataset.map(lambda x: add_max_reward(x,maximal_reward),batched=True)
        
    dataset= dataset.map(formatting_original_dataset_func, batched=True)
   
    results = rollout(
        model,
        tokenizer,
        dataset,
        prompt_field="text",
        generation_args = {
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_new_tokens": 400,
            "top_p": 0.9,
            "stop_string": "</s>"
        },
        batch_size=args.batch_size,
    )
            
    json_res = []
    for question,result,gt,input_prompt in zip(dataset["question"],results,dataset["answer"],dataset["text"]):
        
        splited_res = result["generated_text"].split(ANSWER_TEMPLATE)
        if len(splited_res) > 2:
            completion = ANSWER_TEMPLATE.join(splited_res[1:])
        else:
            completion = splited_res[-1]
            
        json_res.append(
            {
                "input": input_prompt,
                "answer": gt,
                "output": completion,
                "question": question,
                "clean_output": strip_special_tokens(completion,tokenizer=tokenizer),
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