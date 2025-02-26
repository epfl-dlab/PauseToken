# LM Stable Baselines

## Installation

1. Create a new conda envrionment:
    ```
    conda create -n lm_stable_baselines python=3.11
    conda activate lm_stable_baselines
    ````

2. Install lm_stable_baselines:
    ```
    pip install -e .  
    ```
3. Install the rest of the requirements:
    ```
    pip install -r pip_requirements.txt
    ```

## Data

You can download the the data here (it's just the MATH dataset in a special format and gsm8k): https://drive.google.com/file/d/1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340/view?usp=sharing

Once you've downloaded it, unzip it, create a data folder and put it in there. Your data folder should be here:
```
PauseToken
|
|-> data/
    |-> MATH_json/
        |-> train.json
        |-> test.json
    |-> gsm8k_jsonl/
        |-> train.json
        |-> test.json
|-> src/
    |-> model/
    ....
...
```

## Training Curriculum

### Step 1: Warming up model with round of SFT

Here's a template on how to run sft:

```bash
# OPTIONS
## <MODEL-NAME>: mistral, llama1B , llama3B 
## <DATA>: gsm8k , math , pros_qa
## <REWARD>: gsm8k , math , pros_qa
## <METRIC>: gsm8k , math , pros_qa
## <NUM_VAL_SAMPLES>: 748 (for gsm8k), 750 (for math), 300 (for pros_qa)


python src/train.py experiment=train/sft/<MODEL-NAME> data=<DATA> metrics=<METRIC> rl_algorithm/reward=<REWARD> trainer.num_val_samples=<NUM_VAL_SAMPLES> trainer.n_outer_loops=1 run_name=<YOUR-RUN-NAME-HERE>
```

So for example if I want to run sft for mistral on gsm8k:
```bash
python src/train.py experiment=train/sft/mistral data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=1 run_name=sft_mistral_gsm8k
```

For the sake of being fully clear (and because there are a few exceptions) I'll explicitely write all the warm up runs we did

#### All Mistral SFT Runs
```bash
# Mistral on gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/mistral data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=1 run_name=warmup_mistral_gsm8k
# Mistral on Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/mistral data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=1 run_name=warmup_mistral_math rl_algorithm.policy.max_output_generation_length=1800 rl_algorithm.n_envs=16
```

<!-- #### All llama1B SFT Runs
```bash
# llama1B on gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/llama1B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=1 run_name=warmup_llama1B_gsm8k
# llama1B on Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/llama1B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=1 run_name=warmup_llama1B_math rl_algorithm.policy.max_output_generation_length=2048
# llama 1B on pros_qa
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/llama1B data=pros_qa metrics=pros_qa rl_algorithm/reward=pros_qa trainer.num_val_samples=300 trainer.n_outer_loops=1 run_name=warmup_llama1B_pros_qa

``` -->

#### All llama3B SFT Runs
```bash
# llama3B on gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/llama3B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=1 run_name=warmup_llama3B_gsm8k
# llama3B on Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-PATH-TO-MODEL> experiment=train/sft/llama3B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=1 run_name=warmup_llama3B_math rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16
```

#### All Qwen1.5B SFT Runs
```bash
# llama3B on gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=Qwen/Qwen2.5-Math-1.5B experiment=train/sft/qwen1B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=1 run_name=warmup_qwen1B_gsm8k
# llama3B on Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=Qwen/Qwen2.5-Math-1.5B experiment=train/sft/qwen1B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=1 run_name=warmup_qwen1B_math rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16
```

#### All Qwen7B SFT Runs
```bash
# llama3B on gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=Qwen/Qwen2.5-Math-7B experiment=train/sft/qwen7B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=1 run_name=warmup_qwen7B_gsm8k rl_algorithm.n_envs=16
# llama3B on Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=Qwen/Qwen2.5-Math-7B experiment=train/sft/qwen7B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=1 run_name=warmup_qwen7B_math rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16
```


### Step 2: Running Curriculum and the baselines:

Here's a template on how to run Curriculum and it's baseline

```bash
## <MODEL-NAME>: mistral, llama1B , llama3B 
## <DATA>: gsm8k , math , pros_qa
## <REWARD>: gsm8k , math , pros_qa
## <METRIC>: gsm8k , math , pros_qa
## <NUM_VAL_SAMPLES>: 748 (for gsm8k), 750 (for math), 300 (for pros_qa)

# Baseline
python src/train.py experiment=train/ppo/<MODEL-NAME>/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true run_name=<RUN-NAME> rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=<DATA> rl_algorithm/reward=<REWARD> metrics=<METRIC> trainer.num_val_samples=<NUM_VAL_SAMPLES> rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY>

# Curriculum with Beta
python src/train.py experiment=train/ppo/<MODEL-NAME>/curr_beta rl_algorithm.policy.ft_on_action_only=true run_name=<RUN-NAME> rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=10 data=<DATA> rl_algorithm/reward=<REWARD> metrics=<METRIC> trainer.num_val_samples=<NUM_VAL_SAMPLES> rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY>


# Curriculum with Uniform Distribution
python src/train.py experiment=train/ppo/<MODEL-NAME>/curr_uniform rl_algorithm.policy.ft_on_action_only=true run_name=<RUN-NAME> rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=<NUM_VAL_SAMPLES> rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY>
```

For the sake of being perfectly clear, I'll show how to run all 

#### All Mistral Runs

```bash

### PPO Baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/mistral/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=mistral-on-gsm8k run_name=mistral_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/mistral/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=mistral-on-math run_name=mistral_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.policy.max_output_generation_length=1800 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=1800 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=1800


### Curriculum with Beta
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/mistral/curr_beta rl_algorithm.policy.ft_on_action_only=true name=mistral-on-gsm8k run_name=mistral_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/mistral/curr_beta rl_algorithm.policy.ft_on_action_only=true name=mistral-on-math run_name=mistral_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=1800 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=1800


### Curriculum with Uniform
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/mistral/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=mistral-on-gsm8k run_name=mistral_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/mistral/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=mistral-on-math run_name=mistral_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=1800 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=1800


### SFT As baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/mistral data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=50 run_name=sft_mistral_gsm8k
#  Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/mistral data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=50 run_name=sft_mistral_math rl_algorithm.policy.max_output_generation_length=1800 rl_algorithm.n_envs=16
```


#### All LLama3B Runs
```bash

### PPO Baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/llama3B/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=llama3B-on-gsm8k run_name=llama3B_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/llama3B/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=llama3B-on-math run_name=llama3B_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### Curriculum with Beta
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/llama3B/curr_beta rl_algorithm.policy.ft_on_action_only=true name=llama3B-on-gsm8k run_name=llama3B_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/llama3B/curr_beta rl_algorithm.policy.ft_on_action_only=true name=llama3B-on-math run_name=llama3B_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### Curriculum with Uniform
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/llama3B/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=llama3B-on-gsm8k run_name=llama3B_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/llama3B/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=llama3B-on-math run_name=llama3B_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### SFT As baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/llama3B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=50 run_name=sft_llama3B_gsm8k
#  Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/llama3B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=50 run_name=sft_llama3B_math rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16
```


#### All Qwen1B Runs
```bash

### PPO Baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen1B/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=qwen1B-on-gsm8k run_name=qwen1B_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen1B/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=qwen1B-on-math run_name=qwen1B_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### Curriculum with Beta
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen1B/curr_beta rl_algorithm.policy.ft_on_action_only=true name=qwen1B-on-gsm8k run_name=qwen1B_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen1B/curr_beta rl_algorithm.policy.ft_on_action_only=true name=qwen1B-on-math run_name=qwen1B_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### Curriculum with Uniform
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen1B/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=qwen1B-on-gsm8k run_name=qwen1B_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen1B/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=qwen1B-on-math run_name=qwen1B_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### SFT As baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/qwen1B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=50 run_name=sft_qwen1B_gsm8k
#  Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/qwen1B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=50 run_name=sft_qwen1B_math rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16
```

#### All Qwen7B Runs
```bash

### PPO Baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen7B/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=qwen7B-on-gsm8k run_name=qwen7B_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen7B/baseline_sft trainer.n_outer_loops=50 rl_algorithm.policy.ft_on_action_only=true name=qwen7B-on-math run_name=qwen7B_rl_baseline_ppo rl_algorithm.ent_coef=0.009 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### Curriculum with Beta
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen7B/curr_beta rl_algorithm.policy.ft_on_action_only=true name=qwen7B-on-gsm8k run_name=qwen7B_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen7B/curr_beta rl_algorithm.policy.ft_on_action_only=true name=qwen7B-on-math run_name=qwen7B_beta_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.05 trainer.callbacks.portion_annealers.init_beta=5.0 trainer.callbacks.portion_annealers.final_beta=10.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### Curriculum with Uniform
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen7B/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=qwen7B-on-gsm8k run_name=qwen7B_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=gsm8k rl_algorithm/reward=gsm8k metrics=gsm8k trainer.num_val_samples=748
# math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/ppo/qwen7B/curr_uniform rl_algorithm.policy.ft_on_action_only=true name=qwen7B-on-math run_name=qwen7B_unif_curr_ppo rl_algorithm.ent_coef=0.01 rl_algorithm.vf_coef=0.01 rl_algorithm.base_kl_coef=0.01 trainer.n_outer_loops=50 trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 trainer.callbacks.portion_annealers.upper_bound_final_portion=0.0 trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=50 data=math rl_algorithm/reward=math metrics=math trainer.num_val_samples=750 rl_algorithm.n_envs=16 rl_algorithm.n_steps=2 rl_algorithm.batch_size=2 rl_algorithm.n_grad_accumulation_steps=16 rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.policy.model.value_head.transformer_config.config.n_positions=2048


### SFT As baselines ###
# gsm8k
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/qwen7B data=gsm8k metrics=gsm8k rl_algorithm/reward=gsm8k trainer.num_val_samples=748 trainer.n_outer_loops=50 run_name=sft_qwen7B_gsm8k
#  Math
python src/train.py rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH-TO-THE-WARMED-UP-POLICY> experiment=train/sft/qwen7B data=math metrics=math rl_algorithm/reward=math trainer.num_val_samples=750 trainer.n_outer_loops=50 run_name=sft_qwen7B_math rl_algorithm.policy.max_output_generation_length=2048 rl_algorithm.n_envs=16
```

<!-- 2. Installing pytorch. I haven't found a version that works for both on runai and on iccluster 🥲 (something with the `trl` library causes problems)so here's my solution:
    - **If you're on Runai**, install the following requirements for torch:
        ```
        pip install -r runai_torch_requirements.txt
        ```
    - **If you're on the iccluster**, install the following requirements for torch:
        ```
        pip install -r torch_requirements.txt
        ``` -->
<!-- 3. Install the rest of the requirements:
    ```
    pip install -r pip_requirements.txt
    ``` -->


<!-- ## Data Generation

To generate a pause token augmented dataset, you can tweek the following parameters:

- `dataset_location`: The location of the dataset to be augmented.
- `pause_token`: The pause token string to be used for augmentation.
- `n_pauses_per_patterns`:  dictionary of key value pairs where key is the pattern and value is the number of pauses to be injected after an occurence of that pattern"
- `augm_dataset_save_location`: The location where the augmented dataset will be saved.
- `pause_augm_col_name`: The name of the column where the augmented data will be saved in the dataset
- `verbose`: If set, the script will print the progress of the augmentation process.
- `n_random_pauses`: The number of pauses to be injected at random locations (using uniform distribution)
- `tokenizer_hf_name`: The name of the Hugging Face tokenizer to be used to insert random pauses. If None, spaces ' ' will be used to insert random pauses
- `seed`: The seed to be used for random number generation
- `variable_number_of_pauses`: Enable variable number of pauses in sequence (w/ max being n_random_pauses, U[0, n_random_pauses] pauses per sequence)
- `n_generated_samples_per_datapoint`: The number of samples to be generated per datapoint in the dataset (number of y's to be generated per x)

Here is an example of how to use the script with the default parameters:
```bash
python scripts/data_generation/gsm8k_pause_injector.py --dataset_location data/gsm8k_jsonl/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{}' --augm_dataset_save_location data/gsm8k_json/gsm8k_variable_random_pauses --pause_augm_col_name "answer" --verbose --n_random_pauses 100 --tokenizer_hf_name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --variable_number_of_pauses --n_generated_samples_per_datapoint 1 --verbose --seed 42
```

## Train Models (Demo on PauseToken on GSM8K w/ mistral-7B-v0.1)

### Fine-tuning Models

In order to start your RL training with a decent policy, you need to fine-tune your model to randomly insert your control token. Here is an example of how to fine-tune a model on GSM8K dataset with the pause token "<|pause|>":

**Note**: in my directory I have already trained these models. So, you can also use this model to skip the fine-tuning step and directly go to the RL training step. The models locations are:
- <u>Path to Model After Step 1</u>: `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-08-28_13-23-45/final`
- <u>Path to Model After Step 2</u>: INSERT `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-08-28_14-17-32/final`
<!-- - <u>Path to Model After Step 3</u>: INSERT `TODO: INSERT PATH HERE` -->

<!-- #### STEP 1 - Fine-Tune a LM on GSM8K dataset without pause token:
```bash
python src/trl_train.py experiment=trl_train/step_1_sft
```

#### STEP 2 - Augment GSM8K with random pause insertions and train a the pause classifier and pause embedding:
1. **Augment GSM8K on random pause insertions** (see Data Generation section for more details). Here we will augment the GSM8K dataset's ground truth answers with random pauses (ranging from 0 to 100 pauses):
    ```bash
    python scripts/data_generation/gsm8k_pause_injector.py --dataset_location data/gsm8k_jsonl/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{}' --augm_dataset_save_location data/gsm8k_json/gsm8k_variable_random_pauses --pause_augm_col_name "answer" --verbose --n_random_pauses 10 --tokenizer_hf_name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --variable_number_of_pauses --n_generated_samples_per_datapoint 5 --verbose --seed 42
    ```
2. **Fine-tune the model the pause classifier and the pause embedding on GSM8K with the pause token "<|pause|>"**:
2.1. 

    ```bash 
    python src/trl_train.py experiment=trl_train/step_2_sft
    ```
<!-- 
#### STEP 3 - Fine-Tune both the LM and pauseon GSM8K with pause token "<|pause|>":
```bash -->

<!-- ### RL Training Step


#### Onpolicy algorithms (STaR_on_policy, ppo, a2c, etc.)

```
The key parameters to be set are:

trainer.n_outer_loops: # is the number of times an outer loop is called. In each outerloop then the policy.learn(), model validation, and model saving is called respectively.

total_timesteps is the argument for policy.learn(). in LMSB trainer it is set to be equal to inner_loop_timesteps * rl_algorithm.env.num_envs
It defines the number of times we collect rollouts and call policy.train().

trainer.inner_loop_timesteps: # this gets translated into the number of times rollouts are done for each outer loop. 

rl_algorithm:
  n_steps: # used only in collect rollouts. for each environment we callect n_steps and thus the total size of rollout datapoints is n_envs * n_steps
  
  batch_size: # number of rollouts to be sampled and used in each update. has nothing to do with n_envs or anything. 
```



## Experiments

[Click here to see the experiments](./experiments.md)


## Roadmap
- Experiments on On-Policy STaR comparing pause models vs. non-pause models on GSM8K
- Implement sampling of counterfactuals
    - Implement the sampling
    - Determine what loss to use (WSFT ?) (DINA ?)
- Rewards:
    - Implement likelihood of answer reward
        - case 1: compare if answer is correct and if correct give likelihood else give minimum likelihood
        - case 2: manually insert correct answer and give likelihood
- On policy based methods:
    - Reward Conditioning (textual reward or numerical reward)
- Value based methos:
    - Q-learning ([Souce of inspiration](https://github.com/Sea-Snell/Implicit-Language-Q-Learning))
- Actor Critic methods

 -->
