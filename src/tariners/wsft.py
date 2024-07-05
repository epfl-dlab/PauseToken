import pandas as pd
import json
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from datasets import Dataset
import transformers
from transformers import TrainingArguments
from tokenizers import AddedToken
import argparse
import torch
from utils import get_training_args, pause_ground_truth_constrained_rollout, dict_type,strip_special_tokens,count_num_token_occurences
from peft import LoraConfig, TaskType, get_peft_model,AutoPeftModelForCausalLM
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset,concatenate_datasets
import os
import hydra
from rewards import LogLikelihoodReward
import re
from trainers import WeightedSFTTrainer,SFTIgnoreTokensInLossTrainer
from collators import DataCollatorForCompletionOnlyLMWSFT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ANSWER_TEMPLATE = " ### Answer:"
QUESTION_TEMPLATE = " ### Given the following math word problem question generate the correct final answer. Question: "

def wsft_rollout(model, tokenizer, dataset: Dataset, reward, pause_token_id, batch_size, n_samps_per_prompt, max_length, include_gt, pause_temperature) -> Dataset:
    
    def compute_reward_and_text(rollout_result):
     
        generated_text = rollout_result["generated_text"]

        true_reward = reward(
            model_output = generated_text,
            ground_truth = None
        )
        #remove any occurence of <s>
        generated_text = re.sub(r'<s>', '', generated_text)
        return {"text": generated_text,"reward": true_reward, **rollout_result}
        
    def format_prompts_func(example):
        data = []
        for prompt in example['question']:
            data.append(f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{ANSWER_TEMPLATE}")
        return {"input": data}
    
    def prepare_dataset_for_wsft(rollout_res, n_samps_per_prompt,include_gt):
        data = []
        n_samps_per_prompt = n_samps_per_prompt + 1 if include_gt else n_samps_per_prompt
        for i in range(0,len(rollout_res),n_samps_per_prompt):
            data.append(
                {
                    "text": [rollout_res[j]["text"] for j in range(i,i+n_samps_per_prompt)],
                    "rewards": [rollout_res[j]["reward"] for j in range(i,i+n_samps_per_prompt)]
                }
            )
        return data
    
    rollout_dataset = dataset.map(lambda x: format_prompts_func(x), batched=True)
    og_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'     
    rollout_res = pause_ground_truth_constrained_rollout(
        model,
        tokenizer,
        rollout_dataset,
        prompt_field="input",
        ground_truth_field="text",
        pause_token_id=pause_token_id,
        batch_size=batch_size,
        n_samps_per_prompt=n_samps_per_prompt,
        generation_args = {
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_length": max_length,
        },
        include_gt = include_gt,
        pause_temperature = pause_temperature
    )
    
    tokenizer.padding_side = og_padding_side
    reward.set_model(model)
    rollout_res = list(map(compute_reward_and_text, rollout_res))
    #change rollout_res to a list of lists of size n_samps_per_prompt
    rollout_res = prepare_dataset_for_wsft(rollout_res, n_samps_per_prompt, include_gt)
    return Dataset.from_list(rollout_res)
    
def formatting_original_dataset_func(example, task='gsm8k'):
    ## Task = argument or claim
    data = []
    for i in range(len(example['question'])):
        prompt = example['question'][i] #+ " Reasoning Chain: "+ example["predicted_rationale"][i]
        completion = example['answer'][i]
        text = f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{ANSWER_TEMPLATE}{completion.replace('<s>', '')} </s>"
        data.append(text)     
    return {"text": data}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/sft/')
    parser.add_argument('--model-name', default='google/gemma-2b')
    parser.add_argument('--n-epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--batch-size-rollout', default=8, type=int)
    parser.add_argument('--n-samps-per-prompt-rollout', default=1, type=int)
    parser.add_argument('--eval-steps', default=80, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-5, type=float)
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=80, type=int)
    parser.add_argument('--logging-steps', default=80, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--task', required=True) ## arguments or claims
    parser.add_argument('--tag', default='default')
    parser.add_argument('--max-length', default=128, type=int)
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    parser.add_argument('--n-outer-loops', default = 3, type=int)
    parser.add_argument('--modules-to-save', default=[],nargs='*') 
    parser.add_argument('--wsft-beta', default=1.0 , type=float) ## Beta for the weighted SFT
    parser.add_argument('--include-gt', action='store_true') ## Whether to include the ground truth w/out pauses in the rollout
    parser.add_argument('--pause-temperature', default=1.0, type = float) ## 5.3 of Overleaf Amortized Search For Language Model Decoding
    return parser.parse_args()
#python src/DPO/sft.py --model-name=google/gemma-2b --batch-size=16 --use-peft=false

def main():
    args = parse_args()

    model_name = args.model_name
    task = args.task 
    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    input_dir = args.data_dir + task + '/'
    
    if "/" in model_name :
        output_directory =f'{args.output_dir}/{task}/{args.tag}/rcsft_{model_name.split("/")[-1]}_trl_{datetime.now()}'
    else: 
        output_directory =f'{args.output_dir}/{task}/{args.tag}/rcsft_{model_name}_trl_{datetime.now()}'
    args.output_dir = output_directory.replace(' ', '_')
    
    if 't5' in args.model_name.lower(): ### we use T5 but you can use some other model
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    elif 'llama' in args.model_name.lower():
        model = transformers.LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)    
    else: ## if we use Gemma we can just use the AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Add pause token to tokenizer
    pause_token = AddedToken(
        "<|pause|>", 
        single_word=False, 
        lstrip=True, 
        rstrip=True
    )
    tokenizer.add_tokens([pause_token], special_tokens=True)
    tokenizer.pad_token=tokenizer.unk_token
    tokenizer.padding_side = 'right'
    #Resize model to include pause token
    model.resize_token_embeddings(len(tokenizer))
    pause_token_id = tokenizer.convert_tokens_to_ids("<|pause|>")
    
    #load data
    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train').select(range(100))
    #format data
    #Add max reward to dataset
    train_data= train_data.map(formatting_original_dataset_func, batched=True)
    
    rollout_dataset = train_data
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=args.peft_config_r, 
        lora_alpha=args.peft_config_lora_alpha, 
        lora_dropout=args.peft_config_lora_dropout,
        modules_to_save=args.modules_to_save
    )
    
    model = get_peft_model(model, peft_config)
    reward = LogLikelihoodReward(
        tokenizer=tokenizer,
        model=model,
        tokens_ids_to_ignore=[pause_token_id]
    )
    answer_template_ids = tokenizer.encode(ANSWER_TEMPLATE, add_special_tokens=False)[1:]
    
    collator = DataCollatorForCompletionOnlyLM(answer_template_ids, tokenizer=tokenizer)
    wsft_collator = DataCollatorForCompletionOnlyLMWSFT(answer_template_ids, tokenizer=tokenizer)
    
    ## 1 outer loop = WSFT (or SFT if it is the first iteration) + 1 rollout
    for i_outer_loop in range(args.n_outer_loops):
        print(f"OUTER LOOP: {i_outer_loop}")
        training_args = get_training_args(args)
        args.output_dir = f"{output_directory.replace(' ', '_')}_outer_loop_{i_outer_loop}"
        breakpoint()
        #SFT on dataset w/ random pause insertions
        if i_outer_loop == 0:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=rollout_dataset,
                dataset_text_field="text",
                max_seq_length=args.max_length,
                data_collator=collator,
            )
        
        #WSFT on dataset w/ pause insertions (weights based on rewards of Likelihood of sequence excluding pause tokens)
        else:
            training_args.per_device_train_batch_size = max(int(training_args.per_device_train_batch_size/args.n_samps_per_prompt_rollout),1)
            
            trainer = WeightedSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=rollout_dataset,
                dataset_text_field="text",
                max_seq_length=args.max_length,
                data_collator=wsft_collator,
                packing=False,
                n_samples_per_prefix=args.n_samps_per_prompt_rollout if not args.include_gt else args.n_samps_per_prompt_rollout+1,
                reward_col_name="rewards",
                beta=args.wsft_beta
            )

        trainer.train()
        print("SAVING MODEL at ", args.output_dir)
        trainer.save_model(args.output_dir)
        ## Perform rollout    
        rollout_dataset = wsft_rollout(
            model,
            tokenizer,
            train_data,
            reward,
            pause_token_id,
            batch_size=args.batch_size_rollout,
            n_samps_per_prompt=args.n_samps_per_prompt_rollout,
            max_length=args.max_length,
            include_gt=args.include_gt,
            pause_temperature=args.pause_temperature
        )
        #filter out rollouts who have all invalid reward
        min_reward = reward.invalid_ans_penalty
        rollout_dataset = rollout_dataset.filter(lambda x: len(list(filter(lambda y: y > min_reward, x["rewards"]))) > 0)
        rollout_dataset.to_json(f"../data/rollouts/{task}/{args.tag}/{args.output_dir.split('/')[-1]}.json")
    
    ### SUPER UGLY but it works: do last wsft on final rollout
    training_args = get_training_args(args)
    training_args.per_device_train_batch_size = max(int(training_args.per_device_train_batch_size/args.n_samps_per_prompt_rollout),1)
    args.output_dir = f"{output_directory.replace(' ', '_')}_final"
    trainer = WeightedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=rollout_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        data_collator=wsft_collator,
        packing=False,
        n_samples_per_prefix=args.n_samps_per_prompt_rollout if not args.include_gt else args.n_samps_per_prompt_rollout+1,
        reward_col_name="rewards",
        beta=args.wsft_beta
    )
    
    trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
