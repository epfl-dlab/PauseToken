from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from lm_stable_baselines.buffers import LMReplayBuffer, LMRolloutBuffer
import warnings
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from copy import deepcopy
import numpy as np
from typing import Dict
from src.utils.trainer_utils import decode_and_strip_special_tokens, get_aggregated_metrics, decode_and_strip_pad_tokens, save_json, strip_pad_tokens
import shutil
import os
import math
from src.utils.utils import make_summary_table
from transformers.trainer import _is_peft_model
from transformers.utils import ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME

class LMSBTrainer:
    def __init__(
        self,
        rl_algorithm: BaseAlgorithm,
        inner_loop_timesteps: int,
        n_outer_loops: int,
        learn_callbacks: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        progress_bar: bool = False,
        num_val_samples: int = None,
        metrics: Dict = {"test": {}, "val": {}},
        output_dir: str = "output",
        save_top_k: int = 3,
        metric_for_best_model: str = "val/accuracy",
        metric_for_best_model_mode_is_min: bool = False
    ):
        self.learn_kwargs = {
            "total_timesteps": inner_loop_timesteps * rl_algorithm.env.num_envs,
            "callback": learn_callbacks,
            "log_interval": log_interval,
            "tb_log_name": tb_log_name,
            "progress_bar": progress_bar
        }
        self.rl_algorithm = rl_algorithm
        self.n_outer_loops = n_outer_loops
        self.num_val_samples = num_val_samples
        self.logger = self.rl_algorithm.logger
        
        self.current_outer_loop = 0
        self.metrics = metrics
        
        self._train_last_obs = None
        self._val_last_obs = None
        self._train_last_episode_starts = np.ones((self.rl_algorithm.env.num_envs,), dtype=bool)
        self._val_last_episode_starts = np.ones((self.rl_algorithm.env.num_envs,), dtype=bool)
        
        self.output_dir = output_dir
        
        self.save_top_k = save_top_k
        self.metric_for_best_model = metric_for_best_model
        
        self.metric_for_best_model_curr_val = None
        self.metric_for_best_model_mode_is_min = metric_for_best_model_mode_is_min
        self.curr_best_models = []
    
    def set_stage(self, stage: str):
        valid_stages = ["train", "val", "test"]
        assert stage in valid_stages, f"Invalid stage: {stage}, valid stages are: {valid_stages}"
        self.stage = stage        
        
        if stage == "train":
            read_sequentially = False
            self.rl_algorithm.policy.train()
        else:
            read_sequentially = True
            self.rl_algorithm.policy.eval()
        
        self.rl_algorithm.env.set_stage(stage, read_sequentially = read_sequentially)
        
        # Reset env for new stage
        self.rl_algorithm._last_obs = self.rl_algorithm.env.reset()  # type: ignore[assignment]
        self._last_episode_starts = np.ones((self.rl_algorithm.env.num_envs,), dtype=bool)
        # Retrieve unnormalized observation for saving into the buffer
        if self.rl_algorithm._vec_normalize_env is not None:
            self._last_original_obs = self._vec_normalize_env.get_original_obs()
    
    def evaluation(self, stage: str):
        # Run evaluation on validation set 
        
        ################# PART 1: set arguments necessary for performing rollout ################# 
        n_steps = int(math.ceil(self.num_val_samples/self.rl_algorithm.n_envs))
        buffer_name = self.rl_algorithm.buffer_class_keyword
        buffer_name_kwargs = buffer_name + "_kwargs"
        buffer_class = getattr(self.rl_algorithm, buffer_name).__class__
        kwargs = deepcopy(getattr(self.rl_algorithm, buffer_name_kwargs))
        kwargs["advantage_threshold"] = None

        if buffer_class == LMRolloutBuffer:
            buffer_size = n_steps + 1
        elif buffer_class == LMReplayBuffer:
            buffer_size = (n_steps + 1) * self.rl_algorithm.n_envs
        else:
            raise ValueError(f"Invalid buffer class: {buffer_class}, valid buffer classes are: {LMReplayBuffer}, {LMRolloutBuffer}")
        
        validation_buffer = buffer_class(
            buffer_size,
            self.rl_algorithm.observation_space,
            self.rl_algorithm.action_space,
            device = self.rl_algorithm.device,
            n_envs = self.rl_algorithm.n_envs,
            **kwargs
        )
        
        ################# PART 2: perform rollout #################
        #TODO: For the moment, this is fine because 1 step = 1 sample, but in the future, we need to change this for the correct number of samples
        if isinstance(self.rl_algorithm, OffPolicyAlgorithm):
            train_freq = TrainFreq(frequency= n_steps, unit = TrainFrequencyUnit.STEP)

            rollout = self.rl_algorithm.collect_rollouts(
                self.rl_algorithm.env,
                train_freq= train_freq,
                action_noise=self.rl_algorithm.action_noise,
                learning_starts=0,
                replay_buffer=validation_buffer,
                log_interval=self.learn_kwargs["log_interval"],
                callback=self.learn_kwargs["callback"]
            )

            #safety check
            if rollout.n_episodes < self.num_val_samples:
                raise ValueError(
                    f"Expected {self.num_val_samples} samples, but got {rollout.n_episodes} samples, this may be due to the environment not being terminated"
                )

        else:
            rollout = self.rl_algorithm.collect_rollouts(
                self.rl_algorithm.env,
                callback=self.learn_kwargs["callback"],
                rollout_buffer=validation_buffer,
                n_rollout_steps=n_steps+1
            )
            
        ################# PART 3: Collect rollouts from Replay Buffer #################
            
        samps_ids =  np.where(np.ones((self.rl_algorithm.n_envs, n_steps)) == 1)
        samps_ids = (samps_ids[0][:self.num_val_samples], samps_ids[1][:self.num_val_samples])
        val_samps = validation_buffer._get_samples(samps_ids, env = self.rl_algorithm._vec_normalize_env)
        
        next_obs = self.rl_algorithm.get_next_observation(val_samps)
        
        texts = decode_and_strip_pad_tokens(
            next_obs["input_ids"],
            self.rl_algorithm.policy.tokenizer.pad_token_id,
            self.rl_algorithm.policy.tokenizer
        )

        input_texts = decode_and_strip_pad_tokens(
            val_samps.observations["input_ids"],
            self.rl_algorithm.policy.tokenizer.pad_token_id,
            self.rl_algorithm.policy.tokenizer
        )
        
        predicted_outputs = decode_and_strip_pad_tokens(
            val_samps.actions,
            self.rl_algorithm.policy.tokenizer.pad_token_id,
            self.rl_algorithm.policy.tokenizer
        )

        gts = self.rl_algorithm.env.envs[0].get_ground_truths(
            stage=stage,
            idxs = list(range(self.num_val_samples))
        )
        
        reses = []
                
        ################# PART 4: Compute metrics #################
        
        #TODO: Compute or extract metrics (e.g. reward)
        for i,val_samp in enumerate(val_samps.next_observations["input_ids"]):
            
            text = texts[i]
            input_text = input_texts[i]
            predicted_output = predicted_outputs[i]
            
            gt = gts[i]
            tmp_reses = {
                "generated_text": text,
                "tokenized_text": strip_pad_tokens([val_samp.cpu().numpy().tolist()], self.rl_algorithm.policy.tokenizer)[0],
                "input": input_text,
                "predicted_output": predicted_output,
                "ground_truth": gt,
            }
            
            for metric_name, metric_fn in self.metrics[stage].items():
                str_input = decode_and_strip_special_tokens(val_samp, self.rl_algorithm.policy.tokenizer)        
                tmp_reses[metric_name] = metric_fn(str_input, gt)
                
            reses.append(tmp_reses)

        aggregated_metrics = get_aggregated_metrics(reses, list(self.metrics[stage].keys()))
        print(f"Summary Statistics of val:\n {make_summary_table(aggregated_metrics)}")
        if self.metric_for_best_model is not None:
            self.metric_for_best_model_curr_val = aggregated_metrics[f"{self.metric_for_best_model}_mean"]
                        
        ################# PART 5: Save results #################
        
        #TODO: Save validation metrics
        for metric_name, metric_value in aggregated_metrics.items():
            self.rl_algorithm.logger.record(f"{stage}/{metric_name}", metric_value)
            
        #TODO: Save rollouts to file
        save_json(reses, self.output_dir, f"{stage}_results_outer_loop_{self.current_outer_loop}.json")
        
    def run_validation(self):    
        self.evaluation("val")
    
    def run_test(self):
        # Run evaluation on test set
        self.eval_dataset.set_stage("test")
        self.evaluation("test")
    
    def _save_model(self, output_dir, save_type = "lm"):
        save_types = ["lm", "rl_alg"]
        assert save_type in save_types, f"Invalid save_type: {save_type}, valid save_types are: {save_types}"
        
        if save_type == "lm":
            self._lm_save(output_dir)
        else:
            self.rl_algorithm.save(output_dir)
    
    def _lm_save(self, output_dir):
        self.rl_algorithm.policy.lm.save_pretrained(output_dir)
        self.rl_algorithm.policy.tokenizer.save_pretrained(output_dir)
        
    def _lm_load(self, output_dir):
        
        #get class of lm
        class_lm = self.rl_algorithm.policy.lm.__class__
        
        #Inspired by load best model of HF Trainer
        best_adapter_model_path = os.path.join(output_dir, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(output_dir, ADAPTER_SAFE_WEIGHTS_NAME)
        
        
        if _is_peft_model(self.rl_algorithm.policy.lm):
            if (hasattr(self.rl_algorithm.policy.lm, "active_adapter") or hasattr(self.rl_algorithm.policy.lm, "active_adapters")) and hasattr(
                        self.rl_algorithm.policy.lm, "load_adapter"
            ):
                # For BC for older PEFT versions
                if hasattr(self.rl_algorithm.policy.lm, "active_adapters"):
                    active_adapter = self.rl_algorithm.policy.lm.active_adapters[0]
                    if len(self.rl_algorithm.policy.lm.active_adapters) > 1:
                        warnings.warn("Detected multiple active adapters, will only consider the first one")
                else:
                    active_adapter = self.rl_algorithm.policy.lm.active_adapter

                if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                    self.rl_algorithm.policy.lm.load_adapter(output_dir, active_adapter)
                else:
                    warnings.warn(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                warnings.warn("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
            
        else:
            self.rl_algorithm.policy.lm = class_lm.from_pretrained(output_dir)
        
        self.rl_algorithm.policy.tokenizer = self.rl_algorithm.policy.tokenizer.from_pretrained(output_dir)
            
    def _remove_save(self, output_dir):
        try:
            shutil.rmtree(output_dir)
        except:
            warnings.warn(f"Could not remove directory: {output_dir}")
            
    def save_model(self, save_dir = None, save_type = "lm", use_save_top_k = True):
        
        save_types = ["lm", "rl_alg"]
        assert save_type in save_types, f"Invalid save_type: {save_type}, valid save_types are: {save_types}"
        
        passed_invalids_arg_for_save_top_k = False if use_save_top_k and (self.save_top_k is None) else True  
        
        error_message = \
            f"Invalid arguments. If use_save_top_k is True, then self.save_top_k" \
            + f"must be set (and can't be None)." \
            + f" Currently: use_save_top_k: {use_save_top_k}, save_top_k: {self.save_top_k}, "
        
        assert passed_invalids_arg_for_save_top_k, error_message
        
        #if save_dir is None, then we infer the save_dir from the current outer loop (usually it's None if called from fit)
        # save_dir is only not None if the user wants to save the model in a specific directory
        if save_dir is None:
            #number of timesteps taken so far (from learn)
            n_timesteps_taken = self.learn_kwargs["total_timesteps"] * (self.current_outer_loop + 1)
            save_dir = os.path.join(self.output_dir, f"ckpt-{n_timesteps_taken}")
            #append metric to save_dir if it's not None and we want to save the top k models
            save_dir += \
                "" if not use_save_top_k or self.metric_for_best_model is None\
                    else f"-{round(self.metric_for_best_model_curr_val, 3)}" 
        
        #if we only want to save the top k models, then we need to check if the current model is better than the worst model
        if use_save_top_k:
            #if metric_for_best_model is None, then we use the total timesteps taken so far as the metric value
            if self.metric_for_best_model is None:
                self.metric_for_best_model_curr_val = self.learn_kwargs["total_timesteps"] * (self.current_outer_loop + 1)
                self.metric_for_best_model_mode_is_min = False
                
            #append model to list of best models
            self.curr_best_models.append({ "dir": save_dir, "metric_val": self.metric_for_best_model_curr_val})
            #sort models by metric value
            reverse = not self.metric_for_best_model_mode_is_min
            self.curr_best_models = sorted(self.curr_best_models, key = lambda x: x["metric_val"], reverse = reverse)
            
            #If less than save_top_k models, then save the model
            if len(self.curr_best_models) <= self.save_top_k:    
                self._save_model(save_dir, save_type)
            else:
                #get the worst model
                worst_model_path, _ = self.curr_best_models.pop().values()
                #if the worst model is not the same as the current model, then remove the worst model and save the current model
                if worst_model_path != save_dir:
                    self._remove_save(worst_model_path)
                    self._save_model(save_dir, save_type)
        #else, just save the model
        else:
            self._save_model(save_dir, save_type)
            
    def on_validation_start(self):
        self.set_stage("val")
        
         #Probably I need to call predict on the model and collect samples
        if self.num_val_samples is None:
            if "val" in self.rl_algorithm.env.envs[0].dataset:
                warnings.warn("num_val_samples was not provided (None), inferring from dataset")
                num_val_samples = len(self.rl_algorithm.env.envs[0].dataset["val"])
                self.num_val_samples = num_val_samples
            else:
                raise ValueError(f"num_val_samples was not provided (None) and no validation samples were found in the dataset so it could not be inferred")
        
    
    def load_best_model(self):
        if len(self.curr_best_models) == 0:
            raise ValueError("No models have been saved yet")
        reverse = not self.metric_for_best_model_mode_is_min
        self.curr_best_models = sorted(self.curr_best_models, key = lambda x: x["metric_val"], reverse = reverse)
        best_model_path = self.curr_best_models[0]["dir"]
        print("Best model path: ", best_model_path)
        self._lm_load(best_model_path)
    
    def on_validation_end(self):
        self.set_stage("val")
        
    def on_learn_start(self):
        self.set_stage("train")
        
    def on_learn_end(self):
        pass
        
    def on_outer_loop_start(self):
        pass
     
    def fit(self):
        
        for i in range(self.n_outer_loops):
            self.current_outer_loop = i
            self.on_outer_loop_start()
            # Learn
            self.on_learn_start()
            print("Running Learn Stage ... ")
            self.rl_algorithm.learn(**self.learn_kwargs)
            self.on_learn_end()
            
            # Run evaluation on validation set
            self.on_validation_start()
            print("Running Validation Stage ... ")
            self.run_validation()
            self.on_validation_end()
            
            print("Saving model")
            # Save model
            self.save_model()        
            