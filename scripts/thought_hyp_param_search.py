from typing import Any, Dict, List, Optional, Tuple
import hydra
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import torch
from omegaconf import DictConfig,OmegaConf
from pytorch_lightning import seed_everything
from datasets import Dataset
from src.utils.instantiators import instantiate_rl_algorithm, post_instantiation_method_calls, instantiate_model,instantiate_generation_params
from src.model.components.control_token_wrappers import BaseControlTokenWrapper
from tokenizers import AddedToken
from lm_stable_baselines.environments.vectorized_environments import LMDummyVecEnv
from src.utils.utils import make_summary_table
from src.utils.trainer_utils import test_model, save_json
import os
from copy import deepcopy
import math

PERFORMANCE_DECREASE_THRESHOLD_PCT = 0.5
N_SAMPLES = 100
ACCURACY_OF_LOWER_BOUND_IS_ZERO = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOLERANCE = 1e-3

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
    hydra_custom_resolvers,
    make_trainable_params_summary
)


log = RankedLogger(__name__, rank_zero_only=True)


def binary_search_hyperparam(lower_bound, upper_bound, hidden_dim ,cfg, language_model, tokenizer, dataset, generation):

    key_format = "Thought Divisor Exponent = {value}"
    
    
    
    accuracy_table = {}
    if not ACCURACY_OF_LOWER_BOUND_IS_ZERO:
        perf_lower_bound = run_experiment(
            cfg,
            language_model,
            tokenizer,
            dataset,
            generation,
            hidden_dim=hidden_dim,
            thought_head_divisor_exponent=lower_bound,
            save_file_name=f"hyp_seach_{lower_bound}.json"
        )
    else:
        perf_lower_bound = {
            'test/accuracy_mean': 0.0,
        }
    
    lower_performance = perf_lower_bound['test/accuracy_mean']
    print(f"exponent: {lower_bound}, performance: {lower_performance}")
    
    accuracy_table[key_format.format(value=lower_bound)] = perf_lower_bound['test/accuracy_mean']
    
    perf_upper_bound = run_experiment(
        cfg,
        language_model,
        tokenizer,
        dataset,
        generation,
        hidden_dim=hidden_dim,
        thought_head_divisor_exponent=upper_bound,
        save_file_name=f"hyp_seach_{upper_bound}.json"
    )
    
    reference_performance = perf_upper_bound['test/accuracy_mean']
    target_performance = PERFORMANCE_DECREASE_THRESHOLD_PCT * reference_performance
    upper_performance = reference_performance
    print(f"exponent: {upper_bound}, performance: {upper_performance}")

    accuracy_table[key_format.format(value=upper_bound)] = perf_upper_bound['test/accuracy_mean']
    
    while abs(upper_bound - lower_bound) > TOLERANCE:
        mid = (lower_bound + upper_bound) / 2
        
        perf_mid = run_experiment(
            cfg,
            language_model,
            tokenizer,
            dataset,
            generation,
            hidden_dim=hidden_dim,
            thought_head_divisor_exponent=mid,
            save_file_name=f"hyp_seach_{mid}.json"
        )
        
        mid_performance = perf_mid['test/accuracy_mean']
        print(f"exponent: {mid}, performance: {mid_performance}")
        
        accuracy_table[key_format.format(value=mid)] = mid_performance
        
        if mid_performance < target_performance:
            lower_bound = mid
            lower_performance = mid_performance
        else:
            upper_bound = mid
            upper_performance = mid_performance
            
        save_json(accuracy_table, output_folder=cfg.paths.output_dir, file_name="accuracy_table.json")
            
            
    print("exponent search finished")
    print("Summary:")
    print(make_summary_table(accuracy_table))
    
    #save accuracy table in json
    save_json(accuracy_table, output_folder=cfg.paths.output_dir, file_name="accuracy_table.json")
    return accuracy_table

def run_experiment(cfg, language_model, tokenizer, dataset, generation, hidden_dim, thought_head_divisor_exponent, save_file_name):
    
    language_model.thought_embedding_head.divisor_exponent = torch.tensor(thought_head_divisor_exponent, requires_grad=False, device=language_model.thought_embedding_head.hidden_dim.device)
        
    if cfg.get("test_formatting_func"):
        dataset["train"] = dataset["train"].map(
            hydra.utils.instantiate(cfg.test_formatting_func),
            batched=True
        )
        #fetch only the first 1000 samples
    dataset["train"] = dataset["train"].select(range(N_SAMPLES))            
    
    test_metric_fns = {
        f"test/{name}": hydra.utils.get_method(cfg.metrics["test"][name]["_target_"])
        for name in cfg.metrics["test"].keys()
    }
    
    test_summary_metrics = test_model(
        model=language_model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        batch_size=cfg.test_batch_size,
        output_dir=cfg.paths.output_dir,
        prompt_field="input",
        ground_truth_field="output",
        evaluation_metrics=test_metric_fns,
        save_file_name=save_file_name,
        **generation["test"]
    )
    
    log.info(f"Test metrics: {test_summary_metrics}")
    test_metrics = test_summary_metrics
        
    return test_metrics


def load_params(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # torch.autograd.set_detect_anomaly(True)
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.data, _recursive_=False)
    
    log.info(f"Instantiating tokenizer <{cfg.rl_algorithm.policy.model.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.rl_algorithm.policy.model.tokenizer)

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if tokenizer.unk_token is not None:
            log.warning("No padding token found! Setting padding token to unk token.")
            tokenizer.pad_token = tokenizer.unk_token
        else:
            log.warning("No padding token found! To Generation config pad token id")
            pad_token_id = cfg.rl_algorithm.policy.generation.train.generation_config.pad_token_id
            pad_token = tokenizer.decode(pad_token_id)
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = pad_token_id     

    log.info(f"Instantiating language model <{cfg.rl_algorithm.policy.model.language_model._target_}>")
    language_model = instantiate_model(
        cfg.rl_algorithm.policy.model.language_model,
        cfg.rl_algorithm.policy.model.get("peft_config")
    )
    # if cfg.rl_algorithm.policy.get("copy_lm_as_base_model", False):
        # freeze the model and load it
        # base_language_model = deepcopy(language_model)
        # = class_lm.from_pretrained(output_dir, **kwargs).requires_grad_(False)

    # Add control tokens to tokenizer if the language model is a control token wrapper
    if isinstance(language_model, BaseControlTokenWrapper):
        # Add new tokens to tokenizer
        new_tokens = []
        for token_name, token_id in sorted(language_model.config.control_token_to_id.items(), key=lambda x: x[1]):
            new_tokens.append(
                AddedToken(
                    token_name, 
                    single_word=False, 
                    lstrip=True, 
                    rstrip=True
                )
            )
        tokenizer.add_tokens(new_tokens, special_tokens=True)

        #assert that tokenizer token ids match the control token ids
        for token_name, token_id in language_model.config.control_token_to_id.items():
            assert token_id == tokenizer.convert_tokens_to_ids(token_name), \
                f"Token id mismatch for token {token_name}! Expected {token_id} but tokenizer tokenized it as {tokenizer.convert_tokens_to_ids(token_name)}"
    
    # if the language model is predicting thoughts, the eos is going to be shifted too! the generation has to stop
    # when the soft EOS is predicted
    if hasattr(language_model, "thought_mode") and language_model.language_model.config.vocab_size != len(tokenizer):
        tokenizer.set_length(language_model.language_model.config.vocab_size)
    # if hasattr(language_model, "thought_mode") and language_model.thought_mode=='always':
    #     cfg.rl_algorithm.policy.generation.train.generation_config.eos_token_id = tokenizer.eos_token_id + len(tokenizer)
    #     cfg.rl_algorithm.policy.generation.test.generation_config.eos_token_id = tokenizer.eos_token_id + len(tokenizer)

    generation = instantiate_generation_params(
        OmegaConf.to_container(cfg.rl_algorithm.policy.generation,resolve=True)
    )

    language_model.train()
    log.info(f"Summary of model params: \n{make_trainable_params_summary(language_model)}")

    return language_model, tokenizer, dataset, generation


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # log all parameters
    language_model, tokenizer, dataset, generation = load_params(cfg)
    
    language_model = language_model.to(DEVICE)
    
    hidden_dim = float(cfg.rl_algorithm.policy.model.language_model.config.thought_embedding_head.hidden_dim)
    
    accuracy_table = binary_search_hyperparam(
        lower_bound = 0.5,
        upper_bound =  1.0,
        hidden_dim = hidden_dim,
        cfg=cfg,
        language_model=language_model,
        tokenizer=tokenizer,
        dataset=dataset,
        generation=generation,
    )

    # return optimized metric
    return accuracy_table


if __name__ == "__main__":
    main()
