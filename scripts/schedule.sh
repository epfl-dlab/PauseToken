#!/bin/bash

source /dlabscratch1/amani/miniconda3/bin/activate lm_stable_baselines

cd /dlabscratch1/amani/PauseToken/
pwd


python src/train.py experiment=train/ppo/tiny_llama/tiny_llama_pause_bert trainer.n_steps_before_validation=4 \
                                        rl_algorithm.n_steps=1 rl_algorithm.n_envs=4 rl_algorithm.batch_size=4 \
                                        rl_algorithm/reward=gsm8k rl_algorithm.n_grad_accumulation_steps=1


# python src/train.py experiment=train/online_star_exp/pause rl_algorithm.n_steps=9 run_name=online_star_pause_9_step logger.notes="correct nll loss only actions"
# python src/train.py experiment=train/online_star_exp/no_pause_peft rl_algorithm.n_steps=9 run_name=online_star_no_pause_9_step logger.notes="correct nll loss only actions correct val"


# pretraining value head and saving it
# python src/train.py experiment=train/pretraining_value_head/experiment 
# training
# echo "Starting training"

# python src/train.py experiment=/train/online_star_exp/no_pause_peft
# python src/train.py experiment=/train/online_star_exp/pause





# # inference: # For pause models trained on STaR:
# python src/train.py --config-path=/dlabscratch1/amani/PauseToken/logs/train/runs/2024-11-10_11-59-17/.hydra --config-name=config \
# rl_algorithm.policy.model.language_model.pretrained_model_name_or_path='/dlabscratch1/amani/PauseToken/logs/train/runs/2024-11-10_11-59-17/last_ckpt' \
# train=false test=true run_name="star_pause_test" \
# rl_algorithm.policy.generation.generation_config.temperature=1.0 rl_algorithm.policy.generation.generation_config.do_sample=false \
# rl_algorithm.policy.model.peft_config=null