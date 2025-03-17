#!/bin/bash

source /dlabscratch1/amani/miniconda3/bin/activate lm_stable_baselines

cd /dlabscratch1/amani/PauseToken/
pwd

################################################  READ ME  #############################################################
# RL training

# OPTIONS for each dataset and model
## <MODEL-NAME>: mistral, llama1B , llama3B 
## <DATA>: gsm8k , math , pros_qa
## <REWARD>: gsm8k , math , pros_qa
## <METRIC>: gsm8k , math , pros_qa
## <NUM_VAL_SAMPLES>: 748 (for gsm8k), 750 (for math), 300 (for pros_qa)
## <experiment_name>: ppo-on-math, ppo-on-gsm8k, ppo-on-pros_qa

# global options
## ft_on_action_only: true, false
## n_outer_loops: 20, 30
## ent_coef: 0.009, 0.01
## vf_coef: 0.01, 0.1
## base_kl_coef: 0.01, 0.1

# curriculum options, choose one of the three!
# /trainer/callbacks/portion_annealers: null

# /trainer/callbacks/portion_annealers: beta
# trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=10 \
# trainer.callbacks.portion_annealers.init_alpha=5.0 trainer.callbacks.portion_annealers.final_alpha=0.1 \
# trainer.callbacks.portion_annealers.init_beta=50.0 trainer.callbacks.portion_annealers.final_beta=5.0 


# /trainer/callbacks/portion_annealers: uniform
# trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=10 \
# trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 \
# trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 \
# trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 \
# trainer.callbacks.portion_annealers.upper_bound_final_portion=1.0

########################################################################################################################



# Options
EXPERIMENT_PATH=train/ppo/mistral
DATA=math
REWARD=math
METRIC=math
NUM_VAL_SAMPLES=750
EXPERIMENT_NAME=ppo-on-math
NOTE="uniform annealer"

# global options
FT_ON_ACTION_ONLY=true
N_OUTER_LOOPS=20
ENT_COEF=0.009
VF_COEF=0.01
BASE_KL_COEF=0.01
BATCH_SIZE=2
N_GRADIENT_ACCUMULATION=1
N_ENVIRONMENTS=4
# curriculum options, take from readme!



python src/train.py experiment=${EXPERIMENT_PATH} \
name=${EXPERIMENT_NAME} data=${DATA} metrics=${METRIC} rl_algorithm/reward=${REWARD} \
rl_algorithm.ent_coef=${ENT_COEF} rl_algorithm.vf_coef=${VF_COEF} rl_algorithm.base_kl_coef=${BASE_KL_COEF} \
trainer.n_outer_loops=${N_OUTER_LOOPS} trainer.num_val_samples=${NUM_VAL_SAMPLES} logger.notes="${NOTE}" \
rl_algorithm.batch_size=${BATCH_SIZE} rl_algorithm.n_grad_accumulation_steps=${N_GRADIENT_ACCUMULATION} \
rl_algorithm.policy.ft_on_action_only=${FT_ON_ACTION_ONLY} rl_algorithm.n_envs=${N_ENVIRONMENTS} \
trainer/callbacks/portion_annealers=linear \
trainer.callbacks.portion_annealers.warmup_timesteps=0 trainer.callbacks.portion_annealers.total_timesteps=10 \
trainer.callbacks.portion_annealers.lower_bound_init_portion=0.0 \
trainer.callbacks.portion_annealers.lower_bound_final_portion=0.0 \
trainer.callbacks.portion_annealers.upper_bound_init_portion=1.0 \
trainer.callbacks.portion_annealers.upper_bound_final_portion=1.0

# SFT training
# python src/train.py experiment=train/sft/<MODEL-NAME> data=<DATA> metrics=<METRIC> rl_algorithm/reward=<REWARD> 
# trainer.num_val_samples=<NUM_VAL_SAMPLES> trainer.n_outer_loops=1 run_name=<YOUR-RUN-NAME-HERE>
