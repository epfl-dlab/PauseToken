from typing import Any, Dict, List, Optional, Tuple
import hydra
import rootutils
import torch
from omegaconf import DictConfig,OmegaConf
from pytorch_lightning import seed_everything
from datasets import Dataset
from src.utils.instantiators import instantiate_rl_algorithm, post_instantiation_method_calls, instantiate_model,instantiate_generation_params
from src.model.components.control_token_wrappers import BaseControlTokenWrapper
from tokenizers import AddedToken
from lm_stable_baselines.environments.vectorized_environments import LMDummyVecEnv
from src.utils.trainer_utils import test_model
import os

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
    hydra_custom_resolvers,
    make_trainable_params_summary
)


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    if cfg.logger is not None:
        log.info(f"Instantiating up logger <{cfg.logger._target_}>")
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        log.info("No logger found in config! Skipping... Will be using stable-baselines3 logger.")
        logger = None

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.data, _recursive_=False)
    
    log.info(f"Instantiating tokenizer <{cfg.rl_algorithm.policy.model.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.rl_algorithm.policy.model.tokenizer)

    if tokenizer.pad_token is None:
        log.warning("No padding token found! Setting padding token to unk token.")
        tokenizer.pad_token = tokenizer.unk_token
        

    log.info(f"Instantiating language model <{cfg.rl_algorithm.policy.model.language_model._target_}>")
    language_model = instantiate_model(
        cfg.rl_algorithm.policy.model.language_model,
        cfg.rl_algorithm.policy.model.get("peft_config")
    )
    
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
    
    generation = instantiate_generation_params(
        OmegaConf.to_container(cfg.rl_algorithm.policy.generation,resolve=True)
    )
    language_model.train()
    log.info(f"Summary of model params: \n{make_trainable_params_summary(language_model)}")

    log.info(f"Instantiating reward <{cfg.rl_algorithm.reward._target_}>")
    reward = hydra.utils.instantiate(cfg.rl_algorithm.reward, tokenizer=tokenizer)

    log.info(f"instantiating environment <{cfg.rl_algorithm.environment._target_}>")
    env = LMDummyVecEnv(
        [
            lambda: hydra.utils.instantiate(
                cfg.rl_algorithm.environment,
                dataset=dataset,
                tokenizer=tokenizer,
                reward=reward,
                termination_tokens=[tokenizer.eos_token_id],
                n_envs = cfg.rl_algorithm.n_envs, 
                env_idx = i
            )
        for i in range(cfg.rl_algorithm.n_envs)
        ]
    )
    log.info(f"Instantiating RL algorithm <{cfg.rl_algorithm._target_}>")

    rl_alg = instantiate_rl_algorithm(cfg.rl_algorithm, lm=language_model, tokenizer=tokenizer, environment=env, logger=logger)
    log.info(f"Instantiating Trainer <{cfg.trainer._target_}>")
    
    metrics = cfg.get("metrics", {"test": {}, "val": {}})
    metrics_dict = {}
    metrics_dict["test"] = {
        f"test/{name}": hydra.utils.get_method(cfg.metrics["test"][name]["_target_"])
        for name in metrics["test"].keys()
    }
    metrics_dict["val"] = {
        f"val/{name}": hydra.utils.get_method(cfg.metrics["val"][name]["_target_"])
        for name in metrics["val"].keys()
    }
    
    trainer = hydra.utils.instantiate(cfg.trainer, rl_algorithm=rl_alg, metrics=metrics_dict)
    
    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "language_model": language_model,
        "reward": reward,
        "env": env,
        "policy": rl_alg.policy,
        # "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "rl_algorithm": rl_alg,
    }
    
    # TODO: How do we do this ? Sould we just create the wandb logger here?
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit()
        log.info("Training finished! Loading best model...")
        trainer.load_best_model()
        path_to_save = os.path.join(cfg.paths.output_dir, "final")
        trainer.save_model(path_to_save,use_save_top_k=False)
        log.info(f"Saved final model to {cfg.paths.output_dir + '/final'}")
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
            model=trainer.rl_algorithm.policy.lm,
            tokenizer=trainer.rl_algorithm.policy.tokenizer,
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
    metric_dict = { **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
