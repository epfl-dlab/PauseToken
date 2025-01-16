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
from functools import partial
from peft import get_peft_model

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_model(cfg, peft_config=None):
    if not isinstance(cfg, dict):
        model_cfg = OmegaConf.to_container(cfg, resolve=True)
    else:
        model_cfg = copy.deepcopy(cfg)
    
    target_exists = "_target_" in model_cfg 
    method_calls = model_cfg.pop("post_instanciation_method_calls", [])
    
    for key in model_cfg.keys():
        if isinstance(model_cfg[key], dict):
            model_cfg[key] = instantiate_model(model_cfg[key])
    
    if target_exists:     
        model = hydra.utils.instantiate(model_cfg)
        post_instantiation_method_calls(model, method_calls)
        if peft_config is not None:
            peft_config = OmegaConf.to_container(peft_config, resolve=True)
            peft_config = hydra.utils.instantiate(peft_config, _convert_="partial")
            model = get_peft_model(model, peft_config)
        return model
    return cfg    
            
def post_instantiation_method_calls(obj: Any, method_calls: List[Dict[str,Any]]):
    for method_call in method_calls:
        assert "method" in method_call, "Method call must have a 'method' key"
        method = getattr(obj, method_call["method"])
        method = partial(method, *method_call["args"]) if method_call.get("args") else method
        method = partial(method, **method_call["kwargs"]) if method_call.get("kwargs") else method
        method()

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
    
def instantiate_generation_params(cfg):

    if isinstance(cfg, dict):
        for key in cfg.keys():
            cfg[key] = instantiate_generation_params(cfg[key])
            
    cfg = hydra.utils.instantiate(cfg, _convert_="partial") if isinstance(cfg,dict) and "_target_" in cfg else cfg     
    return cfg


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
    buffer_class_keyword = cp.pop("buffer_class_keyword")
    
    cp[buffer_class_keyword+"_class"] = hydra.utils.get_class(cp[buffer_class_keyword+"_class"])
    cp[buffer_class_keyword+"_kwargs"]["tokenizer"] = tokenizer
    cp["policy"] = hydra.utils.get_class(cp.pop("policy_class")) 
    cp["policy_kwargs"]["lm"] = lm
    cp["policy_kwargs"]["tokenizer"] = tokenizer
    cp["env"] = environment

    if "data_collator" in cp:
        data_collator = cp.pop("data_collator")
        add_context_to_response = data_collator.pop("add_context_to_response")

        data_collator["tokenizer"] = tokenizer
        
        #SUPER UGLY BUT I DON'T KNOW HOW TO DO THIS BETTER ##masani: man it's not so bad!
        if "response_template" in data_collator:
            reponse_template = hydra.utils.instantiate(data_collator["response_template"])
            if add_context_to_response:
                reponse_template = "\n" + reponse_template
            response_template_ids = tokenizer.encode(reponse_template, add_special_tokens=False)[2:]
            data_collator["response_template"] = response_template_ids
        data_collator = hydra.utils.instantiate(data_collator)
    
    cp["policy_kwargs"] = {**cp["policy_kwargs"],**{"generation_params": instantiate_generation_params(cp["policy_kwargs"]["generation"])}}
    
    rl_alg = hydra.utils.instantiate(cp, _recursive_=False)
    if not hasattr(rl_alg, "policy"):
        rl_alg._setup_model()
    if logger is not None:
        rl_alg.set_logger(logger)
    rl_alg.data_collator = data_collator
    rl_alg.policy.data_collator = data_collator
    rl_alg.buffer_class_keyword = buffer_class_keyword

    return rl_alg