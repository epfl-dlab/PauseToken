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
from peft import PeftModelForCausalLM
import hashlib
import json
from typing import List

class LMSBTrainer:
    def __init__(
        self,
        rl_algorithm: BaseAlgorithm,
        n_steps_before_validation: int,
        n_outer_loops: int,
        learn_callbacks: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "run",
        progress_bar: bool = False,
        num_val_samples: int = None,
        metrics: Dict = {"test": {}, "val": {}},
        output_dir: str = "output",
        save_top_k: int = 3,
        metric_for_best_model: str = "val/accuracy",
        metric_for_best_model_mode_is_min: bool = False,
        disable_peft_first_inference: bool = False,
        do_sample_at_validation: bool = False,
        peft_config_name: str = "default",
        use_previous_policy_as_reward_model: bool = False,
        idx_of_last_in_context_gt_reasoning_step_distributions: List[int] = None,
    ):
        self.learn_kwargs = {
            "total_timesteps": n_steps_before_validation,
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
        
        self.output_dir = output_dir
        
        self.save_top_k = save_top_k
        self.metric_for_best_model = metric_for_best_model
        
        self.metric_for_best_model_curr_val = None
        self.metric_for_best_model_mode_is_min = metric_for_best_model_mode_is_min
        self.curr_best_models = []
        
        self.disable_peft_first_inference = disable_peft_first_inference
        self.do_sample_at_validation = do_sample_at_validation
        
        self.peft_config_name = peft_config_name
        self.use_previous_policy_as_reward_model = use_previous_policy_as_reward_model
        
        self.idx_of_last_in_context_gt_reasoning_step_distributions = idx_of_last_in_context_gt_reasoning_step_distributions
        
        self.trainer_save_parameters_to_exclude = [
            'learn_kwargs',
            'rl_algorithm',
            'n_outer_loops',
            'num_val_samples',
            'logger',
            'metrics',
            'save_top_k',
            'metric_for_best_model',
            'disable_peft_first_inference',
            'do_sample_at_validation',
            'use_previous_policy_as_reward_model',
            "config_as_string",
            "output_dir",
            "trainer_save_parameters_to_exclude",
            "idx_of_last_in_context_gt_reasoning_step_distributions",
        ]
        
    def set_config_as_string(self, config_as_string: str, name: str, run_name: str):
        self.config_as_string = config_as_string
        
        self.checkpoint_dir = os.path.join(".", "logs", "checkpoints", name,  run_name, self.hash_config())
        
        #make a symbolic link to the output_dir in the checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        output_dir_last_folder_name = os.path.basename(os.path.normpath(self.output_dir))
        
        os.makedirs(os.path.join(self.checkpoint_dir, "previous_runs"), exist_ok=True)
        #get aboslute path of checkpoint_dir
        checkpoint_dir_abs = os.path.abspath(self.checkpoint_dir)
        os.symlink(os.path.abspath(self.output_dir), os.path.join(checkpoint_dir_abs, "previous_runs", output_dir_last_folder_name), target_is_directory=True)

    def hash_config(self):
        return hashlib.sha256(self.config_as_string.encode()).hexdigest()
 
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
        # Retrieve unnormalized observation for saving into the buffer
        if self.rl_algorithm._vec_normalize_env is not None:
            self._last_original_obs = self._vec_normalize_env.get_original_obs()
    
    def evaluation(self, stage: str):
        # Run evaluation on validation set 
        self.rl_algorithm.policy.set_generation_cfg("test")
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

        ################# PART 3: Collect rollouts from Buffers #################
        samps_ids =  np.where(np.ones((n_steps,self.rl_algorithm.n_envs)) == 1)
        samps_ids = (samps_ids[0][:self.num_val_samples], samps_ids[1][:self.num_val_samples])
    
        val_samps = validation_buffer._get_samples(samps_ids, env = self.rl_algorithm._vec_normalize_env)
        val_samps = self.rl_algorithm.process_sampled_rollouts(val_samps) # remove -100 tokens, add 'input_ids' and 'attention_mask'.
        
        if hasattr(val_samps, "next_observations"):
            next_obs = val_samps.next_observations
        else:
            next_obs = self.rl_algorithm.get_next_observation(val_samps)
        
        if isinstance(self.rl_algorithm, OffPolicyAlgorithm):
            mean_reward = val_samps.rewards.mean().item()
        else:
            mean_reward = val_samps.advantages.mean().item()
        
        
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
        for i,val_samp in enumerate(next_obs["input_ids"]):
            
            text = texts[i]
            input_text = input_texts[i]
            predicted_output = predicted_outputs[i]
            
            gt = gts[i]
            tmp_reses = {
                "generated_text": text,
                # "tokenized_text": strip_pad_tokens([val_samp.cpu().numpy().tolist()], self.rl_algorithm.policy.tokenizer)[0],
                "tokenized_text": ', '.join(map(str, strip_pad_tokens([val_samp.cpu().numpy().tolist()], self.rl_algorithm.policy.tokenizer)[0])),
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
        self.rl_algorithm.logger.record(f"{stage}/reward", mean_reward)
        #TODO: Save rollouts to file
        save_json(reses, self.checkpoint_dir, f"{stage}_results_outer_loop_{self.current_outer_loop}.json")

    def run_validation(self):    
        self.evaluation("val")
    
    def run_test(self):
        # Run evaluation on test set
        self.eval_dataset.set_stage("test")
        self.evaluation("test")
    
    
    def save_checkpoint(self):
        path_to_save_rl_alg = os.path.join(self.checkpoint_dir, "last_rl_alg_ckpt.zip")
        path_to_save_trainer = os.path.join(self.checkpoint_dir, "last_trainer_ckpt.zip")
        path_to_policy = os.path.join(self.checkpoint_dir, "last_policy_ckpt.zip")
        path_to_add_mods_policy = os.path.join(self.checkpoint_dir, "last_policy_ckpt")
        
        new_path_to_save_rl_alg = os.path.join(self.checkpoint_dir, f"last_rl_alg_ckpt_2.zip")
        new_path_go_save_trainer = os.path.join(self.checkpoint_dir, f"last_trainer_ckpt_2.zip")
        new_path_to_policy = os.path.join(self.checkpoint_dir, f"last_policy_ckpt_2.zip")
        new_path_to_add_mods_policy = os.path.join(self.checkpoint_dir, "last_policy_ckpt_2")
        
        
        self.save_trainer(self.checkpoint_dir, "last_trainer_ckpt_2.zip")        
        self._save_model(self.checkpoint_dir, save_type="rl_alg", zip_name="last_rl_alg_ckpt_2.zip", policy_name = "last_policy_ckpt_2.zip", exclude=["policy_kwargs"])
        self.rl_algorithm.policy.save_additional_modules(new_path_to_add_mods_policy)
        #remove old files
        for path in [path_to_save_rl_alg, path_to_save_trainer, path_to_policy]:
            if os.path.exists(path):
                self._remove_save(path, is_directory = False)

        if os.path.exists(path_to_add_mods_policy):
            self._remove_save(path_to_add_mods_policy, is_directory = True)

        #rename new files
        os.rename(new_path_to_save_rl_alg, path_to_save_rl_alg)
        os.rename(new_path_go_save_trainer, path_to_save_trainer)
        os.rename(new_path_to_policy, path_to_policy)
        if os.path.exists(new_path_to_add_mods_policy):
            os.rename(new_path_to_add_mods_policy, path_to_add_mods_policy)
    
    def load_checkpoint(self):
        path_to_ckpt_rl_alg = os.path.join(self.checkpoint_dir, "last_rl_alg_ckpt.zip")
        path_to_ckpt_trainer = os.path.join(self.checkpoint_dir, "last_trainer_ckpt.zip")
        path_to_ckpt_policy = os.path.join(self.checkpoint_dir, "last_policy_ckpt.zip")
        path_to_add_mods_policy = os.path.join(self.checkpoint_dir, "last_policy_ckpt")
        
        
        all_ckpt_donot_exist = not os.path.exists(path_to_ckpt_rl_alg) and not os.path.exists(path_to_ckpt_trainer) and not os.path.exists(path_to_ckpt_policy)
        all_ckpt_exist = os.path.exists(path_to_ckpt_rl_alg) and os.path.exists(path_to_ckpt_trainer) and os.path.exists(path_to_ckpt_policy)
        
        assert all_ckpt_donot_exist or all_ckpt_exist, "Both checkpoints must exist or not exist"

        if all_ckpt_exist:
            self._lm_load(os.path.join(self.checkpoint_dir, "last_ckpt"))
            
            self.load_rl_alg(
                self.checkpoint_dir,
                "last_rl_alg_ckpt.zip",
            )
            
            self.load_trainer(path_to_ckpt_trainer)

            self.rl_algorithm.policy.load_additional_modules(path_to_add_mods_policy)
            self.rl_algorithm.policy.to(self.rl_algorithm.device)
            self.load_opt(self.checkpoint_dir, "last_policy_ckpt.zip")
        else:
            print("No checkpoint found, will keep the current state")

        
        
            
    def _save_model(self, output_dir, save_type = "lm", **rl_alg_kwargs):
        save_types = ["lm", "rl_alg"]
        assert save_type in save_types, f"Invalid save_type: {save_type}, valid save_types are: {save_types}"
        
        if save_type == "lm":
            self._lm_save(output_dir)
        else:
            self.rl_algorithm.save(output_dir, **rl_alg_kwargs)
    
    def _lm_save(self, output_dir):
        self.rl_algorithm.policy.lm.save_pretrained(output_dir)
        self.rl_algorithm.policy.tokenizer.save_pretrained(output_dir)
        
    def _lm_load(self, output_dir, force_from_pretrained = False, **kwargs):
        
        #get class of lm
        class_lm = self.rl_algorithm.policy.lm.__class__
        
        #Inspired by load best model of HF Trainer
        best_adapter_model_path = os.path.join(output_dir, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(output_dir, ADAPTER_SAFE_WEIGHTS_NAME)
        
        
        if _is_peft_model(self.rl_algorithm.policy.lm) and not force_from_pretrained:
            if (hasattr(self.rl_algorithm.policy.lm, "active_adapter") or hasattr(self.rl_algorithm.policy.lm, "active_adapters")) and hasattr(
                        self.rl_algorithm.policy.lm, "load_adapter"
            ):
                # For BC for older PEFT versions
                if hasattr(self.rl_algorithm.policy.lm, "active_adapters") and len(self.rl_algorithm.policy.lm.active_adapters) > 0:
                    active_adapter = self.rl_algorithm.policy.lm.active_adapters[0]
                    if len(self.rl_algorithm.policy.lm.active_adapters) > 1:
                        warnings.warn("Detected multiple active adapters, will only consider the first one")
                else:
                    active_adapter = self.rl_algorithm.policy.lm.active_adapter

                if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                    adapter_path = best_adapter_model_path if os.path.exists(best_adapter_model_path) else best_safe_adapter_model_path
                        
                    self.rl_algorithm.policy.lm.unload()
                    self.rl_algorithm.policy.lm = \
                        PeftModelForCausalLM.from_pretrained(self.rl_algorithm.policy.lm.model, model_id = output_dir, adapter_name = "default")
                                        
                    self.rl_algorithm.policy.lm.set_adapter("default")
                else:
                    warnings.warn(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                warnings.warn("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
            
        else:
            self.rl_algorithm.policy.lm = class_lm.from_pretrained(output_dir, **kwargs)
        
        self.rl_algorithm.policy.tokenizer = self.rl_algorithm.policy.tokenizer.from_pretrained(output_dir)
            
    def _remove_save(self, output_dir, is_directory = True):
        try:
            if is_directory:
                shutil.rmtree(output_dir)
            else:
                os.remove(output_dir)
        except:
            warnings.warn(f"Could not remove: {output_dir}")
            
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
            save_dir = os.path.join(self.checkpoint_dir, f"ckpt-{n_timesteps_taken}")
            #append metric to save_dir if it's not None and we want to save the top k models
            save_dir += \
                "" if not use_save_top_k or self.metric_for_best_model is None\
                    else f"-{round(self.metric_for_best_model_curr_val, 3)}"
            
        
        #if we only want to save the top k models, then we need to check if the current model is better than the worst model
        if use_save_top_k:
            last_ckpt = os.path.join(self.checkpoint_dir, f"last_ckpt")
            
            if os.path.exists(last_ckpt):
                self._remove_save(last_ckpt)
            self._save_model(last_ckpt, save_type)
            
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
        
    def load_rl_alg(self, path, zip_name):
        
        self.rl_algorithm.load(
            path=path,
            zip_name= zip_name,
            env = self.rl_algorithm.env,
        )
        
    def load_opt(self, path, policy_name):
        use_base_model_for_learning = getattr(self.rl_algorithm, "use_base_model_for_learning", False)
        print("load optimizer: ", not (self.use_previous_policy_as_reward_model or use_base_model_for_learning))
        if not (self.use_previous_policy_as_reward_model or use_base_model_for_learning):
            self.rl_algorithm.load_optimizer_state_dict(path, policy_name)
        
    def save_trainer(self, path_to_folder, filename):
        trainer_parameters = {k: v for k, v in self.__dict__.items() if k not in self.trainer_save_parameters_to_exclude}
        #save trainer parameters as json
        save_json(trainer_parameters, path_to_folder, filename)
        
    def load_trainer(self, path):
        with open(path, "r") as file:
            trainer_parameters = json.load(file) 
        for k, v in trainer_parameters.items():
            setattr(self, k, v)
    
    def on_validation_end(self):
        use_base_model_for_learning = getattr(self.rl_algorithm, "use_base_model_for_learning", False)
        
        # ~~~ Due to ModulesToSaveWrapper and to not keep all adapters in memory, we need to temporarily save the modules we need, unload all adapters, and reload the modules we need ~~~
        # Why ? Because delete_adapter does not delete the modules to save.... smh
        if use_base_model_for_learning or self.use_previous_policy_as_reward_model:
            
            assert hasattr(self.rl_algorithm.policy.lm, "peft_config"), \
                "We only support use_base_model_for_learning for models that are trained with PEFT (due to memory constraints). If you need this for other models, please implement it."
            
            #delete sampler peft
            if use_base_model_for_learning:
                sampler_name = self.sampler_to_delete
                self.rl_algorithm.policy.lm.delete_adapter(sampler_name)
            #delete reward peft
            elif self.use_previous_policy_as_reward_model and self.name_of_reward_peft != "disable peft":
                reward_name = self.name_of_reward_peft
                if reward_name in self.rl_algorithm.policy.lm.peft_config.keys():
                    self.rl_algorithm.policy.lm.delete_adapter(reward_name)
            
            #set peft to train as adapter (not sure if necessary)
            adapter_name = self.rl_algorithm.name_to_adapter["peft_to_train"]
            self.rl_algorithm.policy.lm.set_adapter(adapter_name)
                        
            #temporarily save the peft model we need
            tmp_save_dir = os.path.join(self.checkpoint_dir, "tmp")
            self._save_model(tmp_save_dir)
            
            #delete adapter and unload all adapters
            self.rl_algorithm.policy.lm.delete_adapter(adapter_name)    
            #unload all adapters. If you're not convinced, put a breakpoint here and you'll see that the ModulesToSaveWrapper of all the adapters we supposedly deleted are still there...
            self.rl_algorithm.policy.lm.unload()
            
            # Reload Model with only peft_to_train adapter and call it default (since now it will be our sampler)
            if use_base_model_for_learning:
                path_to_load = os.path.join(tmp_save_dir, "peft_to_train")
            else:
                path_to_load = tmp_save_dir
            #reload model
            self.rl_algorithm.policy.lm = \
                PeftModelForCausalLM.from_pretrained(self.rl_algorithm.policy.lm.model, model_id = path_to_load, adapter_name = self.peft_config_name, is_trainable=True)
            #remove the temporary save
            self._remove_save(tmp_save_dir)
        
    def on_learn_start(self):
        
        self.rl_algorithm.policy.set_generation_cfg("train")
        is_peft_model = _is_peft_model(self.rl_algorithm.policy.lm)
        # in the first outer loop, option to disable PEFT at inference (Lora is not necessarily trained yet)
        self.rl_algorithm.name_to_adapter = {}
        
        # ~~~~ if we want to sample from the base model on the first outer loop , than do so~~~~ 
        if self.disable_peft_first_inference and is_peft_model and self.current_outer_loop == 0:
            self.rl_algorithm.policy.disable_peft_at_inference()
        else:
            self.rl_algorithm.policy.enable_peft_at_inference()
        
        # ~~~~I we want to always optimize the base model (base model + randomly intialize model with peft), then we need to set the adapter to the base model~~~~
        #check if we want to learn from the initial model as starting point
        use_base_model_for_learning = getattr(self.rl_algorithm, "use_base_model_for_learning", False)
        if use_base_model_for_learning:
            assert hasattr(self.rl_algorithm.policy.lm, "peft_config"), \
                "We only support use_base_model_for_learning for models that are trained with PEFT (due to memory constraints). If you need this for other models, please implement it."
                
            assert len(self.rl_algorithm.policy.lm.peft_config.keys()) >= 1, \
                "We only support use_base_model_for_learning for models that are trained with PEFT (due to memory constraints). If you need this for other models, please implement it."

            
            #get current config name (which should be our sampler). (previous policy)
            cfg_name = self.peft_config_name
            peft_config_cp = deepcopy(self.rl_algorithm.policy.lm.peft_config[cfg_name])
            
            #randomly initialize the model with the peft config (the one to optimize)
            self.rl_algorithm.policy.lm.add_adapter(peft_config = peft_config_cp, adapter_name = "peft_to_train")
            self.rl_algorithm.name_to_adapter = {"peft_to_train": "peft_to_train", "sampler": cfg_name}
            
                        
        # ~~~~ if we want to use the previous policy as the reward model, then we need to set the reward model to the previous policy~~~~
        if self.use_previous_policy_as_reward_model:
            
            #sanity check
            assert hasattr(self.rl_algorithm.policy.lm, "peft_config"), \
                "We only support use_previous_policy_as_reward_model for models that are trained with PEFT (due to memory constraints). If you need this for other models, please implement it."
            
            # if we're in the first loop, then we might as well the disable the peft layer for the reward model
            if self.current_outer_loop == 0:
                name_of_reward_peft = "disable peft"
                self.rl_algorithm.name_to_adapter["peft_to_train"] = self.peft_config_name
            # if we've already loaded the previous policy in use_base_model_for_learning, then we can use the sampler as the reward model (by defintion it's the previous policy)
            elif use_base_model_for_learning:
                name_of_reward_peft = self.rl_algorithm.name_to_adapter["sampler"]
            # otherwise, we need to duplicate the current peft layers and set it as the reward model
            else:
                name_of_reward_peft = "reward"
                
                #temporarily save previous policy
                tmp_save_dir = os.path.join(self.checkpoint_dir, "tmp")
                self._save_model(tmp_save_dir)
                path_to_peft = tmp_save_dir
                possible_path_to_peft = os.path.join(tmp_save_dir, self.peft_config_name)
                if os.path.exists(possible_path_to_peft):
                    path_to_peft = possible_path_to_peft
                
                #set the current peft as the policy to optimize
                self.rl_algorithm.name_to_adapter["peft_to_train"] = self.peft_config_name
                #load the previous policy as the reward model
                self.rl_algorithm.policy.lm.load_adapter(path_to_peft, adapter_name = name_of_reward_peft)
                self._remove_save(tmp_save_dir)
                
            #normally you should have the same lm in all envs so setting the model in one env should be enough
            if hasattr(self.rl_algorithm.env.envs[0].reward, "set_model_peft_name"):
                self.rl_algorithm.env.envs[0].reward.set_model_peft_name(name_of_reward_peft)
            self.name_of_reward_peft = name_of_reward_peft
         
        #rebuild the optimizer (since we've changed the model)
        if self.use_previous_policy_as_reward_model or use_base_model_for_learning:
            self.rl_algorithm.policy.lm.set_adapter(self.rl_algorithm.name_to_adapter["peft_to_train"])
            self.rl_algorithm.policy._build()
        
        self.set_stage("train")
            
    def on_learn_end(self):
        self.rl_algorithm.policy.enable_peft_at_inference()
        use_base_model_for_learning = getattr(self.rl_algorithm, "use_base_model_for_learning", False)
        # For validation, we need to swap the sample with the trained/new policy (it's what we'd like to test)
        if use_base_model_for_learning:
            self.sampler_to_delete = self.rl_algorithm.name_to_adapter["sampler"]
            self.rl_algorithm.name_to_adapter["sampler"] = self.rl_algorithm.name_to_adapter["peft_to_train"]
            
        if hasattr(self.rl_algorithm.policy.lm, "enable_adapter_layers"):
            #just to be sure, enable adapter layers (I've had issues with this in the past)
            self.rl_algorithm.policy.lm.enable_adapter_layers()

    def on_outer_loop_start(self):
        if self.idx_of_last_in_context_gt_reasoning_step_distributions is not None:
            progression = self.current_outer_loop / self.n_outer_loops
            idx_to_chose = int(progression * len(self.idx_of_last_in_context_gt_reasoning_step_distributions))
            distr = self.idx_of_last_in_context_gt_reasoning_step_distributions[idx_to_chose]
            self.rl_algorithm.env.envs[0].update_idx_of_last_in_context_gt_reasoning_step_distr(distr)
        
    def on_outer_loop_end(self):  
        print("Saving model and checkpoint ...")  
        #save lm only
        self.save_model()
        #save_checkpoint (opt, rl_alg)
        self.save_checkpoint()
 
     
    def fit(self):
        self.current_outer_loop = 0
        while self.current_outer_loop < self.n_outer_loops:
            
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
            
            self.current_outer_loop += 1
            self.on_outer_loop_end()
           
