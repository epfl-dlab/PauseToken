import os
from src.utils.trainer_utils import test_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.components.control_token_wrappers import PauseClassifierWrapper
from src.utils.instantiators import instantiate_generation_params
from hydra import initialize, compose
import hydra
from omegaconf import OmegaConf
from typing import List
from src.utils.trainer_utils import inference_formatting_function, reward_conditioning_inference_formatting_function
import json
from datasets import Dataset
from tqdm import tqdm
from functools import partial
from src.utils.constants import CORRECT_ANSWER_FEEDBACK
from torch.cuda import empty_cache, is_available
from torch import bfloat16
import pandas as pd

PATH_TO_DEFAULT_GENERATION_CONFIG = "../../configs/rl_algorithm/policy/generation/"

def load_model_and_tokenizer(name, name_to_path_dict, device_map="auto", use_automodel=True):
    path = name_to_path_dict[name]
    if "no_pause" in name or "baseline" in name or use_automodel:
        model = AutoModelForCausalLM.from_pretrained(path, device_map = device_map)
    elif "pause" or "sft" in name:
        model = PauseClassifierWrapper.from_pretrained(path, torch_dtype=bfloat16)
        #check if gpu is available
    else:
        raise ValueError(f"Unrecognized naming convention for model {name}")
    if is_available() and device_map == "auto":
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    return model, tokenizer

def load_generation_config(pad_token_id: int, eos_token_id: int, bos_token_id: int, max_length: int, overrides: List[str]= []):
    override_token_names = ["generation_config.pad_token_id", "generation_config.eos_token_id", "generation_config.bos_token_id", "generation_config.max_length"]
    overide_token_values = [pad_token_id, eos_token_id, bos_token_id, max_length]
    for name,value in zip(override_token_names, overide_token_values):
        overrides.append(f"{name}={value}")
    
    with initialize(version_base=None, config_path=PATH_TO_DEFAULT_GENERATION_CONFIG):
        cfg = compose(config_name="test_default.yaml", overrides=overrides)
        
    return OmegaConf.to_container(cfg,resolve=True)

def load_test_metrics(metric_config_name: str):
    with initialize(version_base=None, config_path="../../configs/metrics/"):
        cfg = compose(config_name=metric_config_name)
        
    test_metric_fns = {
            f"test/{name}": hydra.utils.get_method(cfg["test"][name]["_target_"])
            for name in cfg["test"].keys()
    }
    return test_metric_fns


def preprocess_data_fn(name, eos_token):
    if "reward_conditioning-" in name:
        return partial(reward_conditioning_inference_formatting_function, eos_token=eos_token, correct_answer_feedback=CORRECT_ANSWER_FEEDBACK)
    else:
        return partial(inference_formatting_function, eos_token=eos_token)

    
    
def load_dataset_from_config(config_name):
    with initialize(version_base=None, config_path="../../configs/data/"):
        cfg = compose(config_name=config_name)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    path = cfg["path"]
    cfg["path"] = os.path.join("..","..", path)
    return hydra.utils.instantiate(cfg, _recursive_=False)

def make_df(results):
    all_res = []
    for model_name, model_results in results.items():
        tmp = list(map(lambda x: {**x, **{"model_name": model_name}}, model_results))
        all_res.extend(tmp)
    return pd.DataFrame(all_res)
        
def rollout_models(
    data_samples: Dataset,
    model_names: List[str],
    generation_config,
    exp_name: str,
    force_overwrite: bool=False,
    batch_size: int=8,
    test_metrics=load_test_metrics("gsm8k"),
    name_to_path_dict: dict = {},
):
    
    results = {}
    output_dir = f"./data/{exp_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pbar =  tqdm(model_names, desc= "Performing rollouts for models", position=1)
    
    for model_name in pbar:
        pbar.set_postfix({'current_model': model_name})
        file_name = f"{model_name}.json"
        path_to_file = os.path.join(output_dir, file_name)
        
        if os.path.exists(path_to_file) and not force_overwrite:
            print(f"File {path_to_file} already exists, skipping generation. Will load from file.")
        else:
            
            model, tokenizer = load_model_and_tokenizer(model_name, name_to_path_dict)
            print("device", model.device)
            preprocess_data = preprocess_data_fn(model_name, tokenizer.eos_token)
            tmp_data_samples = data_samples.map(preprocess_data, batched=True)
            test_model(
                model = model,
                tokenizer = tokenizer,
                dataset = tmp_data_samples,
                batch_size = batch_size,
                output_dir = output_dir,
                prompt_field="input",
                ground_truth_field="output",
                evaluation_metrics = test_metrics,
                save_results=True,
                save_file_name=file_name,
                **generation_config
            )
            #offload model
            del model
            del tokenizer
            #clear gpu
            empty_cache()
            
    
    for model_name in tqdm(model_names, desc="Loading results from files"):
        file_name = f"{model_name}.json"
        path_to_file = os.path.join(output_dir, file_name)
        with open(path_to_file, "r") as f:
            results[model_name] = json.load(f)
            
    return results