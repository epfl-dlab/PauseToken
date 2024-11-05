# Experiments

## Warming up the pause token on random pauses on GSM8K

In this experiment, we try various ways to warm up our model on random pauses on GSM8K. More specifically, we try unfreezing different components of the model. For all of the experiments we start from the same model that can be found at the following path: `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-08-28_13-23-45/final`. The model is fine-tuned on GSM8K for 1 epoch (GSM8K without any pauses). The dataset with random pauses we will use can by found at: `/dlabscratch1/baldwin/pause2/PauseToken/data/gsm8k_json/gsm8k_variable_random_pauses`. Alternatively you can generate it with the following command:
```bash
    python scripts/data_generation/gsm8k_pause_injector.py --dataset_location data/gsm8k_jsonl/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{}' --augm_dataset_save_location data/gsm8k_json/gsm8k_variable_random_pauses --pause_augm_col_name "answer" --verbose --n_random_pauses 10 --tokenizer_hf_name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --variable_number_of_pauses --n_generated_samples_per_datapoint 5 --verbose --seed 42
```

### Training Specifications
All models are trained for 1 epoch on GSM8K with random pauses. The training is done with the following command:

The table below shows for each model the components that are unfrozen. If not specified, all components are frozen.

| experiment-yaml-file                             | Unfreeze Pause Embedding | Unfreeze Pause Head | Unfreeze LM head |  Unfreeze LM Embeddings | LORA | python command                                                                           |
|--------------------------------------------------|:------------------------:|:-------------------:|:----------------:| :----------------------:|:----:|--------------------------------------------------------------------------------------------------------|
| sft.yaml                                         |           X              |      X              |                  |                         |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft.yaml`                                        |
| sft_peft.yaml                                    |           X              |      X              |                  |                         |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_peft.yaml`                                   |
| sft_unfr_lm_head.yaml                            |           X              |      X              |     X            |                         |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_unfr_lm_head.yaml`                           |
| sft_unfr_lm_head_peft                            |           X              |      X              |     X            |                         |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_unfr_lm_head_peft.yaml`                      |
| sft_unfr_lm_head_embed                           |           X              |      X              |     X            |         X               |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_unfr_lm_head_embed.yaml`                     |
| sft_unfr_lm_head_embed_peft                      |           X              |      X              |     X            |         X               |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_unfr_lm_head_embed_peft.yaml`                |
| sft_fr_phead.yaml                                |           X              |                     |                  |                         |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_phead.yaml`                               |
| sft_fr_phead_peft.yaml                           |           X              |                     |                  |                         |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_phead_peft.yaml`                          |
| sft_fr_phead_unfr_lm_head.yaml                   |           X              |                     |     X            |                         |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_phead_unfr_lm_head.yaml`                  |
| sft_fr_phead_unfr_lm_head_peft                   |           X              |                     |     X            |                         |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_phead_unfr_lm_head_peft.yaml`             |
| sft_fr_phead_unfr_lm_head_embed                  |           X              |                     |     X            |         X               |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_phead_unfr_lm_head_embed.yaml`            |
| sft_fr_phead_unfr_lm_head_embed_peft             |           X              |                     |     X            |         X               |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_phead_unfr_lm_head_embed_peft.yaml`       |
| sft_fr_pembed.yaml                               |                          |         X           |                  |                         |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_pembed.yaml`                              |
| sft_fr_pembed_peft.yaml                          |                          |         X           |                  |                         |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_pembed_peft.yaml`                         |
| sft_fr_pembed_unfr_lm_head.yaml                  |                          |         X           |     X            |                         |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_pembed_unfr_lm_head.yaml`                 |
| sft_fr_pembed_unfr_lm_head_peft                  |                          |         X           |     X            |                         |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_pembed_unfr_lm_head_peft.yaml`            |
| sft_fr_pembed_unfr_lm_head_embed                 |                          |         X           |     X            |         X               |      | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_pembed_unfr_lm_head_embed.yaml`           |
| sft_fr_pembed_unfr_lm_head_embed_peft            |                          |         X           |     X            |         X               |  X   | `python src/trl_train.py experiment=trl_train/step2_exp/sft_fr_pembed_unfr_lm_head_embed_peft.yaml`      |
| baseline (model w/out pause; peft) 1 epoch       |                          |                     |                  |                         |   X  | `python src/trl_train.py experiment=trl_train/step_1_sft.yaml`                                           |
| baseline (model w/out pause; peft) 2 epoch       |                          |                     |                  |                         |   X  | `python src/trl_train.py experiment=trl_train/step_1_sft.yaml trainer.args.num_train_epochs=2.0`         |



### Results


| experiment-yaml-file                             |  Test Accuracy | Eval Loss     |  Average Number of pauses per reply | Average Number of pauses per reply (correctly predicted)  | Average Number of pauses per reply (incorrectly predicted) |
|--------------------------------------------------|:--------------:|:-------------:|:------------------------------------:|:--------------------------------------------------------:|:----------------------------------------------------------:|
| sft.yaml                                         |    0.48        |  0.57003      |  1.57                                |                            0.99                          |  2.12                                                      |
| sft_peft.yaml                                    |    **0.56**    |  0.51219      |  1.6                                 |                            1.52                          |  1.7                                                       |
| sft_unfr_lm_head.yaml                            |    0.46        |  0.47131      |  1.33                                |                            0.95                          |  1.64                                                      |
| sft_unfr_lm_head_peft                            |    0.51        |  0.43576      |  5.62                                |                            3.1                           |  8.34                                                      |
| sft_unfr_lm_head_embed                           |    0.49        |  0.40649      |  2.25                                |                            1.86                          |  2.62                                                      |
| sft_unfr_lm_head_embed_peft                      |    0.50        | **0.37507**   | 14.03                                |                            3.37                          | 24.52                                                      |
| sft_fr_phead.yaml                                |    0.14        | 1.99349       |  77.8                                |                           45.39                          | 83.49                                                      |
| sft_fr_phead_peft.yaml                           |    0.05        | 1.96408       | 648.07                               |                           52.18                          | 681.97                                                     |
| sft_fr_phead_unfr_lm_head.yaml                   |    0.10        | 1.92745       | 205.72                               |                           47.72                          | 224.19                                                     |
| sft_fr_phead_unfr_lm_head_peft                   |    0.02        | 2.31768       | 793.02                               |                           51.59                          | 805.59                                                     |
| sft_fr_phead_unfr_lm_head_embed                  |    0.08        | 1.9016        | 365.44                               |                           56.37                          | 391.62                                                     |
| sft_fr_phead_unfr_lm_head_embed_peft             |    0.02        | 2.30675       | 798.37                               |                           55.84                          | 816.24                                                     |
| sft_fr_pembed.yaml                               |    0.46        | 0.77144       | 365.44                               |                           56.37                          | 391.62                                                     |
| sft_fr_pembed_peft.yaml                          |    0.55        | 0.51348       |  1.66                                |                            1.47                          |  1.89                                                      |
| sft_fr_pembed_unfr_lm_head.yaml                  |    0.46        | 0.54759       |  0.77                                |                            0.56                          |  0.95                                                      |
| sft_fr_pembed_unfr_lm_head_peft                  |    0.51        | 0.43743       |  5.02                                |                            3.36                          |  6.77                                                      |
| sft_fr_pembed_unfr_lm_head_embed                 |    0.47        | 0.40834       |  2.06                                |                            1.86                          |  2.24                                                      |
| sft_fr_pembed_unfr_lm_head_embed_peft            |    0.01        | 2.40071       | 781.04                               |                           43.67                          | 787.81                                                     |
| baseline (model w/out pause; peft) 1 epoch       |    0.47        |    -          |  0.0                                 |                            0.0                           |  0.0                                                       |
| baseline (model w/out pause; peft) 2 epoch       |    0.53        |    -          |  0.0                                 |                            0.0                           |  0.0                                                       |


| experiment-yaml-file                       |                               Path to predictions                                             |             Model Location                                                        |                       WandB Link                                     |
|--------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|:--------------------------------------------------------------------:|
|                  sft.yaml                  | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-37-50/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-37-50/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/vm42kwvq) | 
|               sft_peft.yaml                | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-00/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-00/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/71mohe64) |
|           sft_unfr_lm_head.yaml            | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-08/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-08/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/6mou2212) |
|           sft_unfr_lm_head_peft            | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-21/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-21/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/9dl1ri3o) |
|           sft_unfr_lm_head_embed           | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-42-59/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-42-59/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/dbczpbls) |
|        sft_unfr_lm_head_embed_peft         | `dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-27/test_results.json`  | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-27/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/n123pph9) |
|             sft_fr_phead.yaml              | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-51/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-51/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/mrag5uc0) |
|           sft_fr_phead_peft.yaml           | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-56/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-56/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/m2a1s9y7) |
|       sft_fr_phead_unfr_lm_head.yaml       | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-39-17/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-39-17/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/ho8oe4kq) |
|       sft_fr_phead_unfr_lm_head_peft       | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-39-21/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-39-21/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/tobthqyk) |
|      sft_fr_phead_unfr_lm_head_embed       | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-43-29/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-43-29/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/nb7hwpn7) |
|    sft_fr_phead_unfr_lm_head_embed_peft    | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-28/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-28/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/tnfgnwz9) |
|             sft_fr_pembed.yaml             | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-43-29/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-43-29/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/mr3841ax) |
|          sft_fr_pembed_peft.yaml           | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-10/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-10/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/6so42kk0) |
|      sft_fr_pembed_unfr_lm_head.yaml       | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-18/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-18/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/tpiqzrl0) |
|      sft_fr_pembed_unfr_lm_head_peft       | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-26/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-26/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/6com48j8) |
|      sft_fr_pembed_unfr_lm_head_embed      | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-37/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-37/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/xrw3hqo9) |
|   sft_fr_pembed_unfr_lm_head_embed_peft    | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-47/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-47/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/5j8a3kvy) |
|    baseline (model w/out pause; peft) 1    | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-10-53/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-08-28_13-23-45/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/opw5k8mh)                                                       |
| baseline (model w/out pause; peft) 2 epoch | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-08-12/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-08-12/final` | [click here](https://wandb.ai/sigmae/Control%20Tokens/runs/w5xt8w0w) |


### Takeaways

It seems like the best way to 'pretrain' the model on random pauses is by unfreezing the Pause Head, and the Pause embedding and using LoRa for updating the base model. Note however that training the pause embedding doesn't have much of an effect on the accuracy. We do notice a generally tendency (in this pretraining phase), that incorrect replies seem to generate more pauses than incorrect ones on average.


## RL experiments on pause tokens

In this experiment, we compare the training of models on STaR. More specifically, we'll be varying the following axes:
- Model: Pause token model (using the best from previous experiments: `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-00/final`) vs baseline model (no pause token: `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-08-12/final`). For the pause model we'll be training with the best configuration from the previous experiments: `sft_peft.yaml`
- RL algorithm
- temperature (1.0 vs 1.5; see [sampling_playground.ipynb](./notebooks/model_playgrounds/sampling_playground.ipynb) for why we chose these values (TL;DR: at higher temperatures the accuracy of our warmed up models collapses))

### Training Specifications

The table below shows the training specifications for each model. They are all trained for 10 outerloops (10 times rollout on whole dataset + SFT on roullouts).



<!-- possibly depr

| experiment name                                  | Model          | Unfreeze LM Head | STaR alg                                                       | tempearture |             python command                                                                     |
|--------------------------------------------------|:-------------: |:----------------:|:--------------------------------------------------------------:|:-----------:|------------------------------------------------------------------------------------------------|
| offline_star_exp/no_pause_peft                   |  `baseline`    |                  |  `STaR offline`                                                |     1.0     | `python src/train.py experiment=train/offline_star_exp/no_pause_peft`                          |
| offline_star_exp/no_pause_peft_unfr_lm_head      |  `baseline`    |        X         |  `STaR offline`                                                |     1.0     | `python src/train.py experiment=train/offline_star_exp/no_pause_peft_unfr_lm_head`             |
| offline_star_exp/pause                           |  `pause model` |                  |  `STaR offline`                                                |     1.0     | `python src/train.py experiment=train/offline_star_exp/pause`                                  |
| reward_conditioning/no_pause_constant_rc         |  `baseline`    |                  |  `Textual Reward Conditioning (constant text rewards) offline` |     1.0     | `python src/train.py experiment=train/reward_conditioning/no_pause_constant_rc`                |
| reward_conditioning/pause_constant_rc            |  `pause model` |        X         |  `Textual Reward Conditioning (constant text rewards) offline` |     1.0     | `python src/train.py experiment=train/reward_conditioning/pause_constant_rc`                   |
-->

| experiment name                                  | Model          | STaR alg                                                       | temperature |             python command                                                                                                                                                               |
|--------------------------------------------------|:-------------: |:--------------------------------------------------------------:|:-----------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| offline_star_exp/no_pause_peft_temp_1.0          |  `baseline`    |  `STaR offline`                                                |     1.0     | `python src/train.py experiment=train/offline_star_exp/no_pause_peft run_name=offline_star_no_pause_peft_temp_1.0`                                                                       |
| offline_star_exp/pause_temp_1.0                  |  `pause model` |  `STaR offline`                                                |     1.0     | `python src/train.py experiment=train/offline_star_exp/pause run_name=star_pause_temp_1.0`                                                                                               |
| reward_conditioning/no_pause_constant_rc_temp_1.0|  `baseline`    |  `Textual Reward Conditioning (constant text rewards) offline` |     1.0     | `python src/train.py experiment=train/reward_conditioning/no_pause_constant_rc run_name=constant_rc_no_pause_temp_1.0`                                                                   |
| reward_conditioning/pause_constant_rc_temp_1.0   |  `pause model` |  `Textual Reward Conditioning (constant text rewards) offline` |     1.0     | `python src/train.py experiment=train/reward_conditioning/pause_constant_rc run_name=constant_rc_pause_temp_1.0`                                                                          |
| offline_star_exp/no_pause_peft_temp_1.5          |  `baseline`    |  `STaR offline`                                                |     1.5     | `python src/train.py experiment=train/offline_star_exp/no_pause_peft run_name=offline_star_no_pause_peft_temp_1.5 rl_algorithm.policy.generation.generation_config.temperature=1.5`      |
| offline_star_exp/pause_temp_1.5                  |  `pause model` |  `STaR offline`                                                |     1.5     | `python src/train.py experiment=train/offline_star_exp/pause run_name=star_pause_temp_1.5 rl_algorithm.policy.generation.generation_config.temperature=1.5`                              |
| reward_conditioning/no_pause_constant_rc_temp_1.5|  `baseline`    |  `Textual Reward Conditioning (constant text rewards) offline` |     1.5     | `python src/train.py experiment=train/reward_conditioning/no_pause_constant_rc run_name=constant_rc_no_pause_temp_1.5 rl_algorithm.policy.generation.generation_config.temperature=1.5`  |
| reward_conditioning/pause_constant_rc_temp_1.5   |  `pause model` |  `Textual Reward Conditioning (constant text rewards) offline` |     1.5     | `python src/train.py experiment=train/reward_conditioning/pause_constant_rc run_name=constant_rc_pause_temp_1.5 rl_algorithm.policy.generation.generation_config.temperature=1.5`         |

### Results

| experiment name                                  |                               Path to predictions                                             |             Model Location                                                                   |                       WandB Link                                      |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|:--------------------------------------------------------------------: |
| offline_star_exp/no_pause_peft_temp_1.0          |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_09-59-49/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-44-53/final`          | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/kf6quxn2) |
| offline_star_exp/pause_temp_1.0                  |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_10-00-08/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-45-20/final`          | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/sbqi92z0) |
| reward_conditioning/no_pause_constant_rc_temp_1.0|`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_10-10-29/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-45-36/final`          | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/csjcqeer) |
| reward_conditioning/pause_constant_rc_temp_1.0   |`dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_10-01-00/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-45-52/final`          | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/mouzslj9) |                      
| offline_star_exp/no_pause_peft_temp_1.5          |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_10-01-17/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-46-35/last_ckpt`      | [Click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/0yq0dyrp) |
| offline_star_exp/pause_temp_1.5                  |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_10-01-36/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-46-53/last_ckpt`      | [Click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/cnanm007) |
| reward_conditioning/no_pause_constant_rc_temp_1.5|`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_10-02-02/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_17-00-22/last_ckpt`      | [Click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/yc8vizfa) |
| reward_conditioning/pause_constant_rc_temp_1.5   |                                      -                                                        |                                            -                                                 |                                      -                                |       


Collapse of 1.5 temperature models. See their validation predictions for more information because we're sampling from the model at temperature 1.5 (gets better when we resample from argmax): 
-  **offline_star_exp/no_pause_peft_temp_1.5**: `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-46-35/val_results_outer_loop_6.json`
-  **offline_star_exp/pause_temp_1.5**: `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-46-53/val_results_outer_loop_9.json`


| experiment name                                  |  Test Accuracy | Average Number of pauses per reply | Average Number of pauses per reply (correctly predicted)  | Average Number of pauses per reply (incorrectly predicted) |
|--------------------------------------------------|:--------------:|:----------------------------------:|:---------------------------------------------------------:|:----------------------------------------------------------:|
| offline_star_exp/no_pause_peft_temp_1.0          |     0.58       |               0.0                  |                      0.0                                  |                          0.0                               |
| offline_star_exp/pause_temp_1.0                  |     **0.62**   |               2.48                 |                      2.51                                 |                          2.1                               |
| reward_conditioning/no_pause_constant_rc_temp_1.0|     0.49       |               0.0                  |                      0.0                                  |                          0.0                               |
| reward_conditioning/pause_constant_rc_temp_1.0   |     0.49       |               1.0                  |                      1.05                                 |                          0.95                              |
| offline_star_exp/no_pause_peft_temp_1.5          |     0.43       |               0.0                  |                      0.0                                  |                          0.0                               |                                             
| offline_star_exp/pause_temp_1.5                  |     0.45       |               1.4                  |                      0.55                                 |                          2.1                               |
| reward_conditioning/no_pause_constant_rc_temp_1.5|     0.0        |               0.0                  |                      0.0                                  |                          0.0                               |
| reward_conditioning/pause_constant_rc_temp_1.5   |                |



### Important Note
 When running inference, I found that continuing on sampling with the same temperature is not that great and it's better to sample from argmax. In the new code, I've added a flag to sample from argmax at validation and test time. But when I ran these experiments it was not the case so it's possible you see differences in the validation and test results. If you don't manage to get the same results rerun inference with you models. Here's how you can do it for each type of model:
- For non-pause models trained on STaR: 
    ```shell
    python src/train.py experiment=inference/inference_no_pause_model rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH_TO_YOUR_MODEL_HERE> rl_algorithm.policy.generation.generation_config.temperature=1.0 rl_algorithm.policy.generation.generation_config.do_sample=false
    ```
- For pause models trained on STaR: 
    ```shell
    python src/train.py experiment=inference/inference_pause_model rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH_TO_YOUR_MODEL_HERE> rl_algorithm.policy.generation.generation_config.temperature=1.0 rl_algorithm.policy.generation.generation_config.do_sample=false
    ```
- For non-pause models trained on Textual Reward Conditioning: 
    ```shell
    python src/train.py experiment=inference/inference_no_pause_model rc_model.language_model.pretrained_model_name_or_path=<PATH_TO_YOUR_MODEL_HERE> rc_model.generation.generation_config.temperature=1.0 rc_model.generation.generation_config.do_sample=false
    ```
- For non-pause models trained on Constant Textual Reward Conditioning: 
    ```shell
    python src/train.py experiment=inference/inference_no_pause_model_rc rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH_TO_YOUR_MODEL_HERE> rl_algorithm.policy.generation.generation_config.temperature=1.0 rl_algorithm.policy.generation.generation_config.do_sample=false
    ```
- For pause models trained on Constant Textual Reward Conditioning: 
    ```shell
    python src/train.py experiment=inference/inference_pause_model_rc rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=<PATH_TO_YOUR_MODEL_HERE> rl_algorithm.policy.generation.generation_config.temperature=1.0 rl_algorithm.policy.generation.generation_config.do_sample=false
    ```
<!-- Might Depricate -->
<!-- | experiment name                                  |  Test Accuracy | Average Number of pauses per reply | Average Number of pauses per reply (correctly predicted)  | Average Number of pauses per reply (incorrectly predicted) |
|--------------------------------------------------|:--------------:|:----------------------------------:|:---------------------------------------------------------:|:----------------------------------------------------------:|
| offline_star_exp/no_pause_peft.yaml              |    0.53        |               0.0                  |                      0.0                                  |                          0.0                               |
| offline_star_exp/no_pause_peft_unfr_lm_head.yaml |    0.54        |               0.0                  |                      0.0                                  |                          0.0                               |
| offline_star_exp/pause.yaml                      |    0.55        |               2.22                 |                      2.02                                 |                          2.46                              |
| reward_conditioning/no_pause_constant_rc.yaml    |    0.46        |               0.0                  |                      0.0                                  |                          0.0                               |
| reward_conditioning/pause_constant_rc.yaml       |    0.49        |               1.29                 |                      0.71                                 |                          1.86                              |




|                                                  |                               Path to predictions                                             |             Model Location                                                          |                       WandB Link                                      |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------  |:--------------------------------------------------------------------: |
| offline_star_exp/no_pause_peft.yaml              |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-03-19/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-03-19/final` | [part 1](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/0yq0dyrp) [part 2](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/cnanm007) (run crashed so 2 parts)|
| offline_star_exp/no_pause_peft_unfr_lm_head.yaml |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_10-30-30/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-28_11-26-28/final` | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/7um5ztnp) |
| offline_star_exp/pause.yaml                      |`dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-29_09-56-35/test_results.json` | `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-28_11-26-35/final` | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/gigbv2xu) |
| reward_conditioning/no_pause_constant_rc.yaml    |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-03-40/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-03-40/final` | [part 1](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/di9g6j6u) [part 2](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/f0zuv4t9) (run crashed so 2 parts)|
| reward_conditioning/pause_constant_rc.yaml       |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-05-49/test_results.json`| `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-28_11-26-44/final` | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/98w65xd5) | -->


### Takeaways
STaR seems to be the best. Temperature 1.5 from the start might be too agressive. 

## Training Best STaR trained Models till "convergence"

### Training Specifications

From last experiment, we will train the 2 best models till convergence which are: `offline_star_exp/no_pause_peft_temp_1.0` and `offline_star_exp/pause_temp_1.0`. We will keep on training them for 10 more outerloops (10 times rollout on whole dataset + SFT on roullouts). The last checkpoints of these models are `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-44-53/last_ckpt` and `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-45-20/last_ckpt` respectively so we'll start from there. The training specifications are the same as the previous experiment. To resume training, we'll use the following commands:

| experiment name                                  |        python command|
|--------------------------------------------------|-----------------------|
| offline_star_exp/no_pause_peft_temp_1.0_part2    | `python src/train.py experiment=train/offline_star_exp/no_pause_peft rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-44-53/last_ckpt rl_algorithm.policy.model.peft_config=null trainer.n_outer_loops=10 trainer.disable_peft_first_inference=false rl_algorithm/policy/model/language_model=auto_peft_for_causal_lm run_name=offline_star_no_pause_peft_temp_1.0_part2`|
| offline_star_exp/pause_temp_1.0_part2            | `python src/train.py experiment=train/offline_star_exp/pause rl_algorithm.policy.model.language_model.pretrained_model_name_or_path=/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-01_13-45-20/last_ckpt rl_algorithm.policy.model.peft_config=null trainer.n_outer_loops=10 trainer.disable_peft_first_inference=false rl_algorithm/policy/model/language_model=pause_from_pretrained run_name=star_pause_temp_1.0_part2 +rl_algorithm.policy.model.language_model.is_trainable=True`|


### Results

Since I'm trying to get the best model and that my validation accuracy (which is how I chose the best model), I'll be running inference on the test set on the 3 best checkpoints of each model according to the validation accuracy as well as the last checkpoint of each model. The table already presents the results with the best checkpoint of each model.

| experiment name                                  |                               Path to predictions                                             |             Model Location                                                                   |                       WandB Link                                      |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|:--------------------------------------------------------------------: |
|   offline_star_exp/no_pause_peft_temp_1.0_part2  |                                                                                               |                                                                                              | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/9e9bpce2) |
|   offline_star_exp/pause_temp_1.0_part2          |`/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-05_08-30-36/test_results.json`|    `/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-11-04_16-38-59/last_ckpt`   | [click here](https://wandb.ai/sigmae/star%20on%20gsm8k/runs/60ptzzku) |


| experiment name                                  |  Test Accuracy | Average Number of pauses per reply | Average Number of pauses per reply (correctly predicted)  | Average Number of pauses per reply (incorrectly predicted) |
|--------------------------------------------------|:--------------:|:----------------------------------:|:---------------------------------------------------------:|:----------------------------------------------------------:|
| offline_star_exp/no_pause_peft_temp_1.0          |                |                0.0                 |                      0.0                                  |                          0.0                               |
| offline_star_exp/pause_temp_1.0                  |     0.645      |                3.40                |                      3.37                                 |                          3.73                              |