- Chris Suggestion: Force Pause Logit in LM HEAD to Always be -inf even in training x
- Ablation study:
    - freeze lm and run for DPO and Rejection Sampling
    - freeze lm_head and LoRA lm body and run for DPO and Rejection Sampling
    - iLM Style  w/ rejection sampling and DPO 
- Debjit: Introduce incorecness in rejected samples. 

More TODO:
- If still not working, let's try just using DPO Trainers on their own.



Runs to Test:
Testing iLM with dpo_reject no pause Okok
WANDB_MODE=offline python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag tst --modules-to-save embed_tokens lm_head pause_classifier --pause-temperature 1.0 --target-module "q_proj" "v_proj" --filter-if-gt-best-reward --pause-formatting-func "dpo_reject_no_pause" --run-name "tst" --include-gt --debug-num-samples 100

Testing freezing language model body and head + training pause only Okok
WANDB_MODE=offline python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag tst --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_all_pairs" --run-name "tst" --include-gt --debug-num-samples 100 --freeze-lm-body --freeze-lm-head --train-pause-only --include-gt

Testing lm body only. Okok
WANDB_MODE=offline python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag tst --target-modules "q_proj" "v_proj" --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_all_pairs" --run-name "tst" --include-gt --debug-num-s-samples 100 --freeze-lm-head --train-pause-only

Testing ilm Rejection Sampling ?
WANDB_MODE=offline python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag tst --modules-to-save embed_tokens lm_head pause_classifier --pause-temperature 1.0 --target-module "q_proj" "v_proj" --filter-if-gt-best-reward --pause-formatting-func "rejection_sampling_best_sample" --run-name "tst" --include-gt --debug-num-samples 100 --pause-method "pause_rejection_sampling"

Experiments:

Rejection Sampling freezing language model body and head + training pause only 
```
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag rs_freeze_lm --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "rejection_sampling_best_sample" --run-name "rs_freeze_lm" --include-gt --freeze-lm-body --freeze-lm-head --train-pause-only --pause-method "pause_rejection_sampling" --disable-peft
```
Dpo freezing language model body and head + training pause only  w/ dpo_all_pairs
```
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag dpo_freeze_lm_all_pairs --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_all_pairs" --run-name "dpo_freeze_lm_all_pairs" --include-gt --freeze-lm-body --freeze-lm-head --train-pause-only --pause-method "pause_dpo" --disable-peft
````

Dpo freezing language model body and head + training pause only w/ dpo_reject_no_pause 
```
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag dpo_freeze_lm_reject_no_pause --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_reject_no_pause" --run-name "dpo_freeze_lm_reject_no_pause" --include-gt --freeze-lm-body --freeze-lm-head --train-pause-only --pause-method "pause_dpo" --disable-peft


````

Dpo freezing language model head + training pause only w/ dpo_reject_no_pause 
```
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag dpo_freeze_pause_head_reject_no_pause --target-modules "q_proj" "v_proj" --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_reject_no_pause" --run-name "dpo_freeze_pause_head_reject_no_pause" --include-gt  --freeze-lm-head --train-pause-only
```

Dpo freezing language model head + training pause only w/ dpo_all_pairs 
```
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag dpo_freeze_pause_head_all_pairs --target-modules "q_proj" "v_proj" --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_all_pairs" --run-name "dpo_freeze_pause_head_all_pairs" --include-gt  --freeze-lm-head --train-pause-only
```
```
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag rs_freeze_pause_head --target-modules "q_proj" "v_proj" --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "rejection_sampling_best_sample" --run-name "rs_freeze_pause_head" --include-gt  --freeze-lm-head --train-pause-only --pause-method "pause_rejection_sampling" 
```


DO BEST VS WORST SAMPLES freete lm body and head
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag dpo_freeze_lm_reject_best_vs_worst --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_best_vs_worst" --run-name "dpo_freeze_lm_reject_best_vs_worst" --include-gt --freeze-lm-body --freeze-lm-head --train-pause-only --pause-method "pause_dpo" --disable-peft

All pairs till no pause freeze lm body and head
python testing_inv_modeling.py --data-dir "/dlabscratch1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577/" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 3 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 4 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag dpo_freeze_lm_all_pairs_till_no_pauses --modules-to-save pause_classifier --pause-temperature 1.0 --filter-if-gt-best-reward --pause-formatting-func "dpo_all_pairs_till_no_pauses" --run-name "dpo_freeze_lm_all_pairs_till_no_pauses" --include-gt --freeze-lm-body --freeze-lm-head --train-pause-only --pause-method "pause_dpo" --disable-peft



iLM Setups
