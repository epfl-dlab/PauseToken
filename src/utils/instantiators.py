from typing import List, Dict, Any
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from transformers import PreTrainedModel, PreTrainedTokenizer
import hydra
import copy
# from lightning import Callback
# from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig,OmegaConf

from src.utils import pylogger
from omegaconf import open_dict


log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig):
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig):
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
    
def instantiate_rl_algorithm(rl_cfg, lm, tokenizer, environment, logger=None):
    """Instantiates RL algorithm from config.

    :param rl_cfg: A DictConfig object containing RL algorithm configurations.
    :param lm: A PreTrainedModel object.
    :param tokenizer: A PreTrainedTokenizer object.
    :param environment: A LanguageModelEnv object.
    :param logger: A Logger object.
    :return: A BaseAlgorithm object.
    :rtype: BaseAlgorithm
    """
    cp = OmegaConf.to_container(rl_cfg,resolve=True)
    
    
    keys_to_delete = "environment","reward","policy","buffer","n_envs"
    for key in keys_to_delete:
        del cp[key]
    
    cp["replay_buffer_class"] = hydra.utils.get_class(cp["replay_buffer_class"])
    cp["replay_buffer_kwargs"]["tokenizer"] = tokenizer
    cp["policy"] = hydra.utils.get_class(cp.pop("policy_class")) 
    cp["policy_kwargs"]["lm"] = lm
    cp["policy_kwargs"]["tokenizer"] = tokenizer
    cp["env"] = environment
    if cp["policy_kwargs"].get("generation",None) is not None:
        gen_args = cp["policy_kwargs"].pop("generation")
        gen_config = gen_args.pop("generation_config", None)
        gen_config = hydra.utils.instantiate(gen_config) if gen_config is not None else None
        gen_args["generation_config"] = gen_config
        cp["policy_kwargs"] = {**cp["policy_kwargs"], **gen_args}
                
    rl_alg = hydra.utils.instantiate(cp, _recursive_=False)
    if not hasattr(rl_alg, "policy"):
        rl_alg._setup_model()
    if logger is not None:
        rl_alg.set_logger(logger)
    return rl_alg