from typing import Any, Dict, List, Optional, Tuple
import hydra
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from datasets import Dataset
from src.utils.instantiators import instantiate_generation_params,instantiate_model
from src.model.components.control_token_wrappers import BaseControlTokenWrapper
from tokenizers import AddedToken
from lm_stable_baselines.environments.vectorized_environments import LMDummyVecEnv
from src.utils.trainer_utils import test_model
import os

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
    make_trainable_params_summary,
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
    
    # Not great but I don't thing there's a work around for this for HFTrainers
    os.environ["WANDB_PROJECT"] = cfg.name
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
    
    model = instantiate_model(cfg.rl_algorithm.policy.model.language_model)
    
    # Add control tokens to tokenizer if the language model is a control token wrapper
    if isinstance(model, BaseControlTokenWrapper):
        # Add new tokens to tokenizer
        new_tokens = []
        for token_name, token_id in sorted(model.config.control_token_to_id.items(), key=lambda x: x[1]):
            
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
        for token_name, token_id in model.config.control_token_to_id.items():
            assert token_id == tokenizer.convert_tokens_to_ids(token_name), \
                f"Token id mismatch for token {token_name}! Expected {token_id} but tokenizer tokenized it as {tokenizer.convert_tokens_to_ids(token_name)}"
    tokenizer.padding_side = "right"
    log.info(f"Instantiating Trainer <{cfg.trainer._target_}>")
    
    generation = instantiate_generation_params(
        OmegaConf.to_container(cfg.rl_algorithm.policy.generation,resolve=True)
    )

    if cfg.get("save_before_train"):
        model.save_pretrained(cfg.paths.output_dir + "/raw")
        tokenizer.save_pretrained(cfg.paths.output_dir + "/raw")
        log.info(f"Saved model and tokenizer before training to {cfg.paths.output_dir + '/raw'}")
    
    trainer_cfg = OmegaConf.to_container(cfg.trainer,resolve=True)
    if "data_collator" in trainer_cfg:
        trainer_cfg["data_collator"]["tokenizer"] = tokenizer
        #SUPER UGLY BUT I DON'T KNOW HOW TO DO THIS BETTER ##masani: man it's not so bad!
        if "response_template" in trainer_cfg["data_collator"]:
            reponse_template = hydra.utils.instantiate(trainer_cfg["data_collator"]["response_template"])
            response_template_ids = tokenizer.encode(reponse_template, add_special_tokens=False)[1:]
            trainer_cfg["data_collator"]["response_template"] = response_template_ids
    #I have to covert to a container because some of the types canno be save in json (e.g., ListConfig)
    trainer = hydra.utils.instantiate(
        trainer_cfg,
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        eval_dataset=dataset["val"],
        _convert_="partial",
        # label_names=["labels"],
    )
    log.info(f"Summary of model params: \n{make_trainable_params_summary(trainer.model)}")

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
        if cfg.get("merge_peft_after_train") and cfg.trainer.peft_config is not None:
            log.info("Merging PEFT weights with trained weights")
            trainer.model = trainer.model.merge_and_unload()
        trainer.save_model(cfg.paths.output_dir + "/final")
        log.info(f"Saved final model to {cfg.paths.output_dir + '/final'}")
        
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
