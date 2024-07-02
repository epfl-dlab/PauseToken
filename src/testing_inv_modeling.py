import pandas as pd
import json
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
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
from trainers import WeightedSFTTrainerLMLoss,SFTIgnoreTokensInLossTrainer,SFTTrainerPauseLoss,SFTTrainerLMLoss,InvariantModelingTrainer, PauseHeadDPO
from pause_classifier_wrapper import PauseClassifierWrapper, PauseCLFConfig
from callbacks import SwitchLossCallback
from collators import ConditionalCollator,DataCollatorForCompletionOnlyLMWSFT
import copy
os.environ["WANDB_PROJECT"] = "PauseToken"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ANSWER_TEMPLATE = " ### Answer:"
QUESTION_TEMPLATE = " ### Given the following math word problem question generate the correct final answer. Question: "

def ilm_rollout(model, tokenizer, dataset: Dataset, reward, pause_token_id, batch_size, n_samps_per_prompt, max_length, include_gt, pause_temperature) -> Dataset:
    
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
    
    def prepare_ilm_dataset(rollout_res, rollout_dataset ,n_samps_per_prompt,include_gt):
        data = []
        n_samps_per_prompt = n_samps_per_prompt + 1 if include_gt else n_samps_per_prompt
        for i in range(0,len(rollout_res),n_samps_per_prompt):
            prompt = rollout_dataset[i//n_samps_per_prompt]["input"]
            rollout_data = [{"prompt": prompt, "completion": rollout_res[j]["text"].split(ANSWER_TEMPLATE)[1], "reward": rollout_res[j]["reward"]} for j in range(i,i+n_samps_per_prompt)]
            rollout_data = sorted(rollout_data, key=lambda x: x["reward"], reverse=True)
            rollout_data = \
                {
                    "prompts": [x["prompt"] for x in rollout_data],
                    "completions": [x["completion"] for x in rollout_data], 
                    "rewards": [x["reward"] for x in rollout_data],
                }
            data.append(rollout_data)
        return data
    was_in_training_mode = model.training
    model.eval()
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
    )
    tokenizer.padding_side = og_padding_side
    reward.set_model(model)
    rollout_res = list(map(compute_reward_and_text, rollout_res))
    rollout_res = prepare_ilm_dataset(rollout_res, rollout_dataset, n_samps_per_prompt, include_gt)
    if was_in_training_mode:
        model.train()
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

def get_dpo_formating_func(func_type):
    def dpo_all_pairs(example):
        data = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(example["prompts"])):
            prompts = example["prompts"][i]
            completions = example["completions"][i]
            
            for j in range(len(completions)):
                chosen_completion = completions[j]
                for k in range(j+1,len(completions)):
                    reject_completion = completions[k]
                    data["prompt"].append(prompts[j])
                    data["chosen"].append(chosen_completion)
                    data["rejected"].append(reject_completion)

        return data
    def dpo_best_vs_worst(example):
        data = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(example["prompts"])):
            prompts = example["prompts"][i]
            completions = example["completions"][i]
            best_completion = completions[0]
            worst_completion = completions[-1]
            data["prompt"].append(prompts[0])
            data["chosen"].append(best_completion)
            data["rejected"].append(worst_completion)
        return data
    
    def dpo_all_pairs_till_no_pauses(example):
        data = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(example["prompts"])):
            prompts = example["prompts"][i]
            completions = example["completions"][i]
            #find the first completion without a pause
            no_pauses_idx = None
            for j in range(len(completions)):
                if "<|pause|>" not in completions[j]:
                    no_pauses_idx = j
                    break
            if no_pauses_idx is None or no_pauses_idx == 0:
                continue
            for j in range(no_pauses_idx):
                chosen_completion = completions[j]
                for k in range(j+1,no_pauses_idx+1):
                    reject_completion = completions[k]
                    data["prompt"].append(prompts[j])
                    data["chosen"].append(chosen_completion)
                    data["rejected"].append(reject_completion)
            
        return data
    
    def dpo_reject_no_pause(example):
        data = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(example["prompts"])):
            prompts = example["prompts"][i]
            completions = example["completions"][i]
            #find the first completion without a pause
            no_pauses_idx = None
            for j in range(len(completions)):
                if "<|pause|>" not in completions[j]:
                    no_pauses_idx = j
                    break
            if no_pauses_idx is None or no_pauses_idx == 0:
                continue
            rejected = completions[no_pauses_idx]
            for j in range(no_pauses_idx):
                chosen_completion = completions[j]
                data["prompt"].append(prompts[j])
                data["chosen"].append(chosen_completion)
                data["rejected"].append(rejected)
            
        return data
    
    valid_func_types = ["dpo_all_pairs", "dpo_best_vs_worst", "dpo_all_pairs_till_no_pauses","dpo_reject_no_pause"]
    assert func_type in valid_func_types, f"Invalid DPO formatting function, valid types are {valid_func_types}"

    if func_type == "dpo_all_pairs":
        return dpo_all_pairs
    elif func_type == "dpo_best_vs_worst":
        return dpo_best_vs_worst
    elif func_type == "dpo_all_pairs_till_no_pauses":
        return dpo_all_pairs_till_no_pauses
    elif func_type == "dpo_reject_no_pause":
        return dpo_reject_no_pause
        
def get_pause_rejection_sampling_formating_func(func_type):
    def rejection_sampling_best_sample(example):
        data = []
        for i in range(len(example["prompts"])):
            prompt = example["prompts"][i][0]
            best_completion = example["completions"][i][0]
            text = f"{prompt[0]}{ANSWER_TEMPLATE}{best_completion}"
            data.append(text)
        return {"text": data}
    valid_func_types = ["rejection_sampling_best_sample"]
    assert func_type in valid_func_types, f"Invalid pause rejection sampling formatting function, valid types are {valid_func_types}"
    if func_type == "rejection_sampling_best_sample":
        return rejection_sampling_best_sample
    
def sft_formating_func(example):
    data = []
    for i in range(len(example["prompts"])):
        prompts = example["prompts"][i]
        completions = example["completions"][i]
        for j in range(len(prompts)):
            text = f"{prompts[j]}{ANSWER_TEMPLATE}{completions[j]}"
            data.append(text)
    return {"text": data}

def wsft_formatting_func(example):
    data = {"text": [], "rewards": []}
    for i in range(len(example["prompts"])):
        prompts = example["prompts"][i]
        completions = example["completions"][i]
        texts = []
        rewards = []
        for j in range(len(prompts)):
            texts.append(f"{prompts[j]}{ANSWER_TEMPLATE}{completions[j]}")
            rewards.append(example["rewards"][j])
        data["text"].append(texts)
        data["rewards"].append(rewards)
    return data


def get_ilm_trainer(model, tokenizer, training_args, args, rollout_dataset ,pause_token_id ,pause_method = None, pretrain_phase = False ):
    
    assert pause_method in ["pause_rejection_sampling", "pause_dpo","pause_sft"] or pause_method == None, "Invalid pause_method"
    answer_template_ids = tokenizer.encode(ANSWER_TEMPLATE, add_special_tokens=False)[1:]
    completion_only_collator = DataCollatorForCompletionOnlyLM(answer_template_ids, tokenizer=tokenizer)   
    
    num_to_trainer_config = {}
    name_to_formatting_func = {}
    num_to_train_method = {}
    name_to_collator = {}
    
    train_methods = []
    train_methods.append(pause_method)
    if pause_method == "pause_sft":
         
        trainer_pause_loss_config = \
            {
                "trainer_class": SFTTrainerPauseLoss,
                "model": model,
                "tokenizer": tokenizer,
                "args": training_args,
                "max_seq_length": args.max_length,
                "dataset_text_field": "text",
            }
            
        collator = completion_only_collator
        formatting_func = None
        
        
        
    elif pause_method == "pause_dpo":
        
        trainer_pause_loss_config = \
            {
                "trainer_class": PauseHeadDPO,
                "pause_token_id": pause_token_id,
                "model": model,
                "tokenizer": tokenizer,
                "args": training_args,
            }
        collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id, label_pad_token_id=-100, is_encoder_decoder=False)
        
        formatting_func = get_dpo_formating_func(args.pause_formatting_func)
        
        
    else:
        trainer_pause_loss_config = \
            {
                "trainer_class": SFTTrainerPauseLoss,
                "model": model,
                "tokenizer": tokenizer,
                "args": training_args,
                "max_seq_length": args.max_length,
                "dataset_text_field": "text",
            }
        collator = completion_only_collator
        
        formatting_func = get_pause_rejection_sampling_formating_func(args.pause_formatting_func)
        
        
        
    num_to_trainer_config[0] = trainer_pause_loss_config
    num_to_train_method[0] = pause_method
    name_to_collator[0] = collator
    name_to_formatting_func[pause_method] = formatting_func
    
    if not args.train_pause_only:
        valid_lm_methods = ["lm_head_sft","lm_head_wsft"] 
        assert args.lm_method in valid_lm_methods, "Invalid lm_method"
        if args.lm_method == "lm_head_sft" or pretrain_phase:
            lm_method = "lm_head_sft"
            trainer_lm_loss_config = \
                {
                    "trainer_class":  SFTTrainerLMLoss,
                    "model": model,
                    "tokenizer": tokenizer,
                    "args": training_args,
                    "max_seq_length": args.max_length,
                    "dataset_text_field": "text",
                }
            collator = completion_only_collator
            
            formatting_func = None if pretrain_phase else sft_formating_func
            
        elif args.lm_method == "lm_head_wsft":
            lm_method = "lm_head_wsft"
            trainer_lm_loss_config = \
                {
                    "trainer_class":  WeightedSFTTrainerLMLoss,
                    "model": model,
                    "tokenizer": tokenizer,
                    "args": training_args,
                    "max_seq_length": args.max_length,
                    "dataset_text_field": "text",
                    "packing": False,
                    "n_samples_per_prefix": args.n_samps_per_prompt_rollout if not args.include_gt else args.n_samps_per_prompt_rollout+1,
                    "reward_col_name": "rewards",
                    "beta": 0.1,
                    "pause_token_id": pause_token_id,
                }
            
            collator = DataCollatorForCompletionOnlyLMWSFT(answer_template_ids, tokenizer=tokenizer)

            formatting_func = wsft_formatting_func
        
        num_to_trainer_config[1] = trainer_lm_loss_config
        num_to_train_method[1] = lm_method
        name_to_collator[1] = collator
        name_to_formatting_func[lm_method] = formatting_func
        train_methods.append(lm_method)

    rollout_dataset = rollout_dataset.map(
                lambda x: add_train_methods(x, train_methods),
            )
    
    ilm_collator = ConditionalCollator(name_to_collator= name_to_collator, name_col="train_method")
    
    columns_to_keep = ["text", "train_method"] if pretrain_phase else ["prompts", "completions", "rewards", "train_method"]
    
    trainer = InvariantModelingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=rollout_dataset,
        data_collator=ilm_collator,
        num_to_train_method = num_to_train_method,
        columns_to_keep=columns_to_keep,
        num_to_trainer_config=num_to_trainer_config,
        name_to_formatting_func=name_to_formatting_func,
        trainer_name_col="train_method",
    )
    return trainer
    
def add_train_methods(example, train_methods):
    example["train_method"] = train_methods
    return example


def filter_if_gt_best_reward(example):
    best_sample = example["completions"][0]
    if "<|pause|>" in best_sample:
        return True
    return False



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
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
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
    parser.add_argument('--target-modules', default=[], nargs='*')
    parser.add_argument('--include-gt', action='store_true') ## Whether to include the ground truth w/out pauses in the rollout
    parser.add_argument('--pause-temperature', default=1.0, type = float) ## 5.3 of Overleaf Amortized Search For Language Model Decoding
    parser.add_argument('--disable-peft', action="store_true") ## Whether to disable PEFT
    parser.add_argument('--run-name', default='default')
    parser.add_argument("--filter-if-gt-best-reward", action="store_true")
    parser.add_argument("--is-pause-pretrained", action="store_true")
    parser.add_argument("--train-pause-only", action="store_true")
    parser.add_argument("--freeze-lm-head", action="store_true")
    parser.add_argument("--freeze-lm-body", action="store_true")
    parser.add_argument("--pause-method", default ="pause_dpo")
    parser.add_argument("--pause-formatting-func", default="dpo_reject_no_pause")
    parser.add_argument("--debug-num-samples", default= -1, type=int)
    parser.add_argument("--first-rollout-ds-path", default=None, type=str)
    parser.add_argument("--lm-method", default="lm_head_sft")
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
        output_directory =f'{args.output_dir}/{task}/{args.tag}/ilm_{model_name.split("/")[-1]}_trl_{datetime.now()}'
    else: 
        output_directory =f'{args.output_dir}/{task}/{args.tag}/ilm_{model_name}_trl_{datetime.now()}'
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
    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train')
    if args.debug_num_samples > 0:
        train_data = train_data.select(range(args.debug_num_samples))
    #format data
    #Add max reward to dataset
    train_data= train_data.map(formatting_original_dataset_func, batched=True)
    
    rollout_dataset = train_data

    pause_clf_config = PauseCLFConfig(
        pause_token_id=pause_token_id,
        loss_type="pause_loss",
    )
    model = PauseClassifierWrapper(pause_clf_config,model)
    
    if args.freeze_lm_head:
        model.freeze_lm_head()
    if args.freeze_lm_body:
        model.freeze_lm_body()
    
    model.set_pause_temperature(args.pause_temperature)
    if not args.disable_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.peft_config_r, 
            lora_alpha=args.peft_config_lora_alpha, 
            lora_dropout=args.peft_config_lora_dropout,
            modules_to_save=args.modules_to_save,
            target_modules= args.target_modules
        )
        if not args.is_pause_pretrained:
            model = get_peft_model(model, peft_config)

    reward = LogLikelihoodReward(
        tokenizer=tokenizer,
        model=model,
        tokens_ids_to_ignore=[pause_token_id]
    )
    
    training_args = get_training_args(args)
    args.output_dir = f"{output_directory.replace(' ', '_')}_sft"
    # training_args.run_name = f"{args.run_name}_sft"
    
    if not args.is_pause_pretrained:
        
        trainer = get_ilm_trainer(
            model = model,
            tokenizer = tokenizer,
            training_args = training_args,
            args = args,
            rollout_dataset = rollout_dataset,
            pause_token_id = pause_token_id,
            pause_method = "pause_sft",
            pretrain_phase = True
        )
        
        #print allocated and reserved memory
        trainer.train()
        print("SAVING MODEL at ", args.output_dir)
        if not args.disable_peft:
            #Merge previous LoRA
            trainer.model = trainer.model.merge_and_unload()
            trainer.save_model(args.output_dir)
            #Create New Lora
            model = trainer.model
        else:
            trainer.save_model(args.output_dir)
    
    ## 1 outer loop = WSFT (or SFT if it is the first iteration) + 1 rollout
    for i_outer_loop in range(args.n_outer_loops):
        print(f"OUTER LOOP: {i_outer_loop}")
        training_args = get_training_args(args)
        args.output_dir = f"{output_directory.replace(' ', '_')}_outer_loop_{i_outer_loop}"
        # training_args.run_name = f"{args.run_name}_outer_loop_{i_outer_loop}"
        ## Perform rollout
        if i_outer_loop == 0 and args.first_rollout_ds_path is not None:
            rollout_dataset = Dataset.from_json(args.first_rollout_ds_path)
        else:
            rollout_dataset = ilm_rollout(
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
            rollout_dataset.to_json(f"../data/rollouts/{task}/{args.tag}/{args.output_dir.split('/')[-1]}.json")
        
        if args.filter_if_gt_best_reward:
            og_len = len(rollout_dataset)
            rollout_dataset = rollout_dataset.filter(filter_if_gt_best_reward)
            print(f"Filtered out {og_len - len(rollout_dataset)} samples, {len(rollout_dataset)} out of {og_len} remain")
        
        if not args.disable_peft:
            model = get_peft_model(model, peft_config)
            
        trainer = get_ilm_trainer(
            model = model,
            tokenizer = tokenizer,
            training_args = training_args,
            args = args,
            rollout_dataset = rollout_dataset,
            pause_token_id = pause_token_id,
            pause_method = args.pause_method,
            pretrain_phase = False,
        )
        # if args.train_pause_only:
        #     trainer_pause_loss_config = \
        #     {
        #         "trainer_class": PauseHeadDPO,
        #         "pause_token_id": pause_token_id,
        #         "model": model,
        #         "tokenizer": tokenizer,
        #         "args": training_args,
        #     }
            
            
            
        # else:
        #     trainer_pause_loss_config = \
        #     {
        #         "trainer_class": PauseHeadDPO,
        #         "pause_token_id": pause_token_id,
        #         "model": model,
        #         "tokenizer": tokenizer,
        #         "args": training_args,
        #     }
        #     trainer_lm_loss_config = \
        #     {
        #         "trainer_class":  SFTTrainerLMLoss,
        #         "model": model,
        #         "tokenizer": tokenizer,
        #         "args": training_args,
        #         "max_seq_length": args.max_length,
        #         "dataset_text_field": "text",
        #     }
            
        #     num_to_trainer_config = {
        #         0: trainer_pause_loss_config,
        #         1: trainer_lm_loss_config        
        #     }
        #     name_to_formatting_func = {
        #         "pause_dpo": dpo_formating_func,
        #         "lm_head_sft": sft_formating_func
        #     }
        #     num_to_train_method = {
        #         0: "pause_dpo",
        #         1: "lm_head_sft"
        #     }
        

        #     dpo_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id, label_pad_token_id=-100, is_encoder_decoder=False)
            
        #     ilm_collator = ConditionalCollator(name_to_collator= {0: dpo_collator, 1: collator}, name_col="train_method")

        # #repeat each sample of rollout_dataset twice (roullout dataset should be 2x the size of the original dataset)
        # # rollout_dataset = rollout_dataset.map(duplicate_dataset, batched=True, batch_size=training_args.per_device_train_batch_size)
        # #filter out rollouts who have all invalid reward
        # # min_reward = reward.invalid_ans_penalty
        # # rollout_dataset = rollout_dataset.filter(lambda x:  x["reward"]> min_reward)
        

        
        # #SFT on dataset w/ random pause insertions
        # trainer = InvariantModelingTrainer(
        #     model=model,
        #     tokenizer=tokenizer,
        #     args=training_args,
        #     train_dataset=rollout_dataset,
        #     data_collator=ilm_collator,
        #     num_to_train_method = num_to_train_method,
        #     columns_to_keep=["prompts", "completions", "rewards", "train_method"],
        #     num_to_trainer_config=num_to_trainer_config,
        #     name_to_formatting_func=name_to_formatting_func,
        #     trainer_name_col="train_method",
        # )


        trainer.train()
        print("SAVING MODEL at ", args.output_dir)
        if not args.disable_peft:
            #Merge previous LoRA
            trainer.model = trainer.model.merge_and_unload()
            trainer.save_model(args.output_dir)
            #Create New Lora
            model = trainer.model
        else:
            trainer.save_model(args.output_dir)
        
        
if __name__ == '__main__':
    main()