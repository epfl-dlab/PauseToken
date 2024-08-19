
from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from datasets import Dataset
from src.utils.instantiators import instantiate_generation_params
from lm_stable_baselines.environments.vectorized_environments import LMDummyVecEnv
from src.utils.trainer_utils import test_model
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    hydra_custom_resolvers
)
from trl import SFTTrainer

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def trl_train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
        
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.data, _recursive_=False)

    log.info(f"Instantiating tokenizer <{cfg.rl_algorithm.policy.model.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.rl_algorithm.policy.model.tokenizer)

    if tokenizer.pad_token is None:
        log.warning("No padding token found! Setting padding token to unk token.")
        tokenizer.pad_token = tokenizer.unk_token
        
    log.info(f"Instantiating language model <{cfg.rl_algorithm.policy.model.language_model._target_}>")
    model = hydra.utils.instantiate(cfg.rl_algorithm.policy.model.language_model)
    
    log.info(f"Instantiating Trainer <{cfg.trainer._target_}>")
    
    generation = instantiate_generation_params(
        OmegaConf.to_container(cfg.rl_algorithm.policy.generation,resolve=True)
    )
    
    #I have to covert to a container because some of the types canno be save in json (e.g., ListConfig)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        eval_dataset=dataset["val"],
        _convert_="partial",
    )
    
    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "model": model,
        "trainer": trainer,
    }

    train_metrics = {}
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.train()
        train_metrics = trainer.state.__dict__
        trainer.save_model(ksjlfkjsdlkf)
        # trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        
    test_metrics = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        
        if cfg.get("test_formatting_func"):
            dataset["test"] = dataset["test"].map(
                hydra.utils.instantiate(cfg.test_formatting_func),
                batched=True
            )
        
        test_metric_fns = {
            f"test/{name}": hydra.utils.get_method(cfg.metrics["test"][name]["_target_"])
            for name in cfg.metrics["test"].keys()
        }
        
        test_summary_metrics = test_model(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=dataset["test"],
            batch_size=cfg.test_batch_size,
            output_dir=cfg.paths.output_dir,
            prompt_field="input",
            ground_truth_field="output",
            evaluation_metrics=test_metric_fns,
            **generation
        )
        
        log.info(f"Test metrics: {test_summary_metrics}")
        test_metrics = test_summary_metrics
        

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="trl_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = trl_train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()



# ANSWER_TEMPLATE = " ### Answer:"
# QUESTION_TEMPLATE = " ### Given the following math word problem question generate the correct final answer. Question: "


    
# def formatting_original_dataset_func(example, task='gsm8k'):
#     ## Task = argument or claim
#     data = []
#     for i in range(len(example['question'])):
#         prompt = example['question'][i] #+ " Reasoning Chain: "+ example["predicted_rationale"][i]
#         completion = example['answer'][i]
#         text = f" [INST]{QUESTION_TEMPLATE}{prompt} [/INST]\n\n{ANSWER_TEMPLATE}{completion.replace('<s>', '')} </s>"
#         data.append(text)     
#     return {"text": data}


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', default='data/sft/')
#     parser.add_argument('--model-name', default='google/gemma-2b')
#     parser.add_argument('--n-epochs', default=1, type=int)
#     parser.add_argument('--batch-size', default=8, type=int)
#     parser.add_argument('--batch-size-rollout', default=8, type=int)
#     parser.add_argument('--n-samps-per-prompt-rollout', default=1, type=int)
#     parser.add_argument('--eval-steps', default=80, type=int)
#     parser.add_argument('--eval-batch-size', default=32, type=int)
#     parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
#     parser.add_argument('--learning-rate', default=2e-5, type=float)
#     parser.add_argument('--warmup-steps', default=0, type=int)
#     parser.add_argument('--warmup-ratio', default=0.0, type=float)
#     parser.add_argument('--weight-decay', default=0.01, type=float)
#     parser.add_argument('--adam-epsilon', default=1e-8, type=float)
#     parser.add_argument('--save-steps', default=80, type=int)
#     parser.add_argument('--logging-steps', default=80, type=int)
#     parser.add_argument('--output-dir', default='models')
#     parser.add_argument('--task', required=True) ## arguments or claims
#     parser.add_argument('--tag', default='default')
#     parser.add_argument('--max-length', default=128, type=int)
#     parser.add_argument('--peft-config-r', default=16, type=int)
#     parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
#     parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
#     parser.add_argument('--n-outer-loops', default = 3, type=int)
#     parser.add_argument('--modules-to-save', default=[],nargs='*')
#     parser.add_argument('--target-modules', default=[], nargs='*')
#     parser.add_argument('--run-name', default='default')
#     parser.add_argument("--debug-num-samples", default= -1, type=int)
#     return parser.parse_args()

# def main():
    
    
#     if args.data_dir[-1] != '/':
#         args.data_dir += '/'

#     input_dir = args.data_dir + task + '/'
    
#     if "/" in model_name :
#         output_directory =f'{args.output_dir}/{task}/{args.tag}/ilm_{model_name.split("/")[-1]}_trl_{datetime.now()}'
#     else: 
#         output_directory =f'{args.output_dir}/{task}/{args.tag}/ilm_{model_name}_trl_{datetime.now()}'
#     args.output_dir = output_directory.replace(' ', '_')
    
#     if 't5' in args.model_name.lower(): ### we use T5 but you can use some other model
#         model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
#         tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
#     elif 'llama' in args.model_name.lower():
#         model = transformers.LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype=torch.bfloat16, device_map="auto")
#         tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)    
#     else: ## if we use Gemma we can just use the AutoModelForCausalLM
#         model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token=tokenizer.unk_token
#     tokenizer.padding_side = 'right'
#     #load data
#     train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train')
    
    
#     if args.debug_num_samples > 0:
#         train_data = train_data.select(range(args.debug_num_samples))
#     #split into train and validation with seed 123
#     train_data= train_data.map(formatting_original_dataset_func, batched=True)
#     data = train_data.train_test_split(test_size=0.1, seed=123)
#     train_data = data['train']
#     test_data = data['test']
    
#     #format data
#     #Add max reward to dataset
    
      
#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM, 
#         inference_mode=False, 
#         r=args.peft_config_r, 
#         lora_alpha=args.peft_config_lora_alpha, 
#         lora_dropout=args.peft_config_lora_dropout,
#         modules_to_save=args.modules_to_save,
#         target_modules= args.target_modules
#     )
    
#     model = get_peft_model(model, peft_config)

#     training_args = get_training_args(args)
#     training_args.evaluation_strategy = "epoch"
#     training_args.save_strategy = "epoch"
#     training_args.eval_steps = 1
#     training_args.do_eval = True
#     training_args.load_best_model_at_end=True
#     training_args.save_total_limit = 10
    
#     answer_template_ids = tokenizer.encode(ANSWER_TEMPLATE, add_special_tokens=False)[1:]
        
#     completion_only_collator = DataCollatorForCompletionOnlyLM(answer_template_ids, tokenizer=tokenizer)
    
#     trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         max_seq_length=args.max_length,
#         dataset_text_field="text",
#         train_dataset=train_data,
#         data_collator=completion_only_collator,
#         eval_dataset=test_data,
#     )
        
    
#     trainer.train()
#     print("SAVING MODEL at ", args.output_dir)
    
    
# if __name__ == '__main__':
#     main()