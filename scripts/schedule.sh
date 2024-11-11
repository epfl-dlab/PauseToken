#!/bin/bash

source /dlabscratch1/amani/miniconda3/bin/activate lm_stable_baselines

cd /dlabscratch1/amani/PauseToken/
pwd


# training
# echo "Starting training"

# python src/train.py experiment=/train/online_star_exp/no_pause_peft
python src/train.py experiment=/train/online_star_exp/pause





# # inference: # For pause models trained on STaR:
# python src/train.py --config-path=/dlabscratch1/amani/PauseToken/logs/train/runs/2024-11-10_11-59-17/.hydra --config-name=config \
# rl_algorithm.policy.model.language_model.pretrained_model_name_or_path='/dlabscratch1/amani/PauseToken/logs/train/runs/2024-11-10_11-59-17/last_ckpt' \
# train=false test=true run_name="star_pause_test" \
# rl_algorithm.policy.generation.generation_config.temperature=1.0 rl_algorithm.policy.generation.generation_config.do_sample=false