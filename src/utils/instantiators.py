from typing import List, Dict, Any
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from transformers import PreTrainedModel, PreTrainedTokenizer
import hydra
import copy
# from lightning import Callback
# from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def instantiate_language_model(policy_cfg: DictConfig) -> PreTrainedModel:
    """Instantiates policies from config.

    :param policy_cfg: A DictConfig object containing policy configurations.
    :type policy_cfg: DictConfig
    :return: A Language Model.
    :rtype: PreTrainedModel
    """
    lm = hydra.utils.instantiate(policy_cfg.lm, _recursive_=False)
    return lm

def instantiate_tokenizer(policy_cfg: DictConfig) -> PreTrainedTokenizer:
    """Instantiates policies from config.

    :param policy_cfg: A DictConfig object containing policy configurations.
    :type policy_cfg: DictConfig
    :return: A tokenizer
    :rtype: PreTrainedTokenizer
    """
    tokenizer = hydra.utils.instantiate(policy_cfg.tokenizer, _recursive_=False)
    return tokenizer

def prepare_policy_args(policy_cfg: DictConfig) -> Dict[str, Any]:
    """Instantiates policies from config.

    :param policy_cfg: A DictConfig object containing policy configurations.
    :type policy_cfg: DictConfig
    :return: A dictionary containing policy arguments.
    :rtype: Dict[str, Any]
    """
    lm = instantiate_language_model(policy_cfg)
    tokenizer = instantiate_tokenizer(policy_cfg)
    
    policy_cfg = dict(policy_cfg)
    
    policy_cfg["lm"] = lm
    policy_cfg["tokenizer"] = tokenizer
    
    target = policy_cfg.pop("_target_")
    
    return {"policy": target, "policy_kwargs": policy_cfg}

def prepare_buffer_args(buffer_cfg: DictConfig) -> Dict[str, Any]:
    buffer_cfg = dict(buffer_cfg)
    target = buffer_cfg.pop("_target_")
    return {"buffer_class": target, "buffer_kwargs": buffer_cfg}
    

def instantiate_rl_algorithm(rl_alg_cfg: DictConfig) -> BaseAlgorithm:
    """Instantiates policies from config.

    :param policy_cfg: A DictConfig object containing policy configurations.
    :type policy_cfg: DictConfig
    :return: A Language Model.
    :rtype: PreTrainedModel
    """
    
    cfg = copy.deepcopy(rl_alg_cfg)
    policy_args = prepare_policy_args(cfg.policy)
    buffer_args = prepare_buffer_args(cfg.buffer)
    cfg = dict(cfg)
    
    cfg = {**cfg, **policy_args}
    cfg = {**cfg, **buffer_args}
    #TODO: instantiate rl_algo (Optionally add callbacks too)
    #retrun hydra.utils.instantiate(cfg)
    
    
    
    

