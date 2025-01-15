# Experiments

## Stage 1: Warming up Models on GSM8K

In this, stage we will be warming up both pause and non-pause models on GSM8K dataset. To create a dataset with random pauses you can use the following command:

```bash
 python scripts/data_generation/gsm8k_probabilistic_pause_injector.py --dataset_location data/gsm8k_jsonl/gsm8k --pause_token "<|pause|>" --augm_dataset_save_location data/gsm8k_json/gsm8k_pause_prob0.1-5samp --pause_augm_col_name "answer" --tokenizer_hf_name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --verbose --seed 42 --n_generated_samples_per_datapoint 5
```

### Warming up Mistral-7B on GSM8K

#### Training Specifications

| Experiment Name                   | Unfreeze Pause Embedding | Unfreeze Pause Head | Unfreeze LM head |  Unfreeze LM Embeddings | LORA  |  Detach Pause Head     |   python command                                                               |
|-----------------------------------|:------------------------:|:-------------------:|:----------------:| :----------------------:|:----: |------------------------|--------------------------------------------------------------------------------|
| Warmup Mistral Pause detached     |           X              |      X              |                  |                         |   X   |           X            |  `python src/trl_train.py  experiment=trl_train/sft_pause_detached run_name=warmup_mistral_pause_detached_5samp_ds_mean_kl data=gsm8k_augmented_pause trainer.args.num_train_epochs=1.0`
       |
| Warmup Mistral No-Pause           |                          |                     |                  |                         |   X   |           X            |  `python experiment=trl_train/step_1_sft.yaml trainer.args.num_train_epochs=5.0 run_name=step1_sft_2epochs`
               |
| Warmup Mistral Pause              |           X              |      X              |                  |                         |   X   |                        |  `python src/trl_train.py experiment=trl_train/sft_pause run_name=warmup_mistral_pause_5samp_ds_mean_kl data=gsm8k_augmented_pause trainer.args.num_train_epochs=1.0`           |





### Warming up TinyLlama-1B on GSM8K

#### Training Specifications

| Experiment Name                                     | Train Full Model | Pause Model  |  Detach Pause Head    |  Pause Head Training Method           | python command                                                                                                                                                                  |
|-----------------------------------------------------|:----------------:|:------------:|:---------------------:|:-------------------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| warmup_tinyllama_pause_detached KL on true probs    |       X          |     X        |           X           | KL on true probabilites (0.1 % pause) | `python src/trl_train.py experiment=trl_train/sft_pause_tiny_llama_pause_detached run_name=warmup_tinyllama_pause_detached_5samp_ds data=gsm8k_augmented_pause trainer.args.num_train_epochs=1.0` |
| warmup_tinyllama_pause KL on true probs             |       X          |     X        |                       | KL on true probabilites (0.1 % pause) |`python src/trl_train.py experiment=trl_train/sft_pause_tiny_llama_pause run_name=warmup_tinyllama_pause_5samp_ds data=gsm8k_augmented_pause trainer.args.num_train_epochs=1.0`|
| warmup_tinyllama_pause KL on one-hot probs          |       X          |              |                       | KL on one hot probabilites            |`python src/trl_train.py experiment=trl_train/sft_pause_tiny_llama_pause run_name=warmup_tinyllama_pause_5samp_ds_og_trainer data=gsm8k_augmented_pause trainer.args.num_train_epochs=1.0 trainer._target_=trl.SFTTrainer ~trainer.pause_probability`|                                                                                  
| warmup_tinyllama_pause_detached KL on one-hoe probs |       X          |     X        |           X           | KL on one hot probabilites            |`python src/trl_train.py experiment=trl_train/sft_pause_tiny_llama_pause_detached run_name=warmup_tinyllama_pause_detached_5samp_ds_og_trainer data=gsm8k_augmented_pause trainer.args.num_train_epochs=1.0 trainer._target_=trl.SFTTrainer ~trainer.pause_probability`|
| warmup tinyllama no pause (baseline)                |       X          |              |           N/A          | N/A                                   |`python src/trl_train.py experiment=trl_train/sft_tiny_llama run_name=warmup_tinyllama`|
#### Results

| Experiment Name                                   |  Test Accuracy | Average Pause per reply | Average Number of pauses per reply (correctly predicted)  | Average Number of pauses per reply (incorrectly predicted) | Percentage of pause tokens in predictions |
|---------------------------------------------------|:-------------:|:-----------------------:|:---------------------------------------------------------:|:-----------------------------------------------------------:|-------------------------------------------|
|warmup_tinyllama_pause_detached KL on true probs   | **0.0584**     |       16.27             |                         **12.96**                         |                           16.47                             |                   0.11                    |
|warmup_tinyllama_pause KL on true probs            | 0.0553        |       **17.2**          |                         12.03                             |                           **17.51**                         |                   0.10                    |
|warmup_tinyllama_pause KL on one-hot probs         | 0.0311        |       13.98             |                         10.39                             |                           14.09                             |                   0.07                    |
|warmup_tinyllama_pause_detached KL on one-hot probs| 0.0295        |       17.02             |                         10.67                             |                           17.21                             |                   0.09                    |
|Warmup tinyllama no pause (baseline)               | 0.0531        |       0.0               |                         0.0                               |                           0.0                               |                   0.0                     |


| Experiment Name                                   |                               Path to predictions                                             |             Model Location                                                           |                       WandB Link                         |
|---------------------------------------------------|:------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| warmup_tinyllama_pause_detached KL on true probs  |`/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-47-16/test_results.json`     | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-47-16/final` | `https://wandb.ai/sigmae/Control%20Tokens/runs/0g24tsll` |
| warmup_tinyllama_pause KL on true probs           |`/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-45-38/test_results.json`     | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-45-38/final` | `https://wandb.ai/sigmae/Control%20Tokens/runs/qaqedgv6` |
| warmup_tinyllama_pause KL on one-hot probs        |`/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-24-57/test_results.json`     | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-24-57/final` | `https://wandb.ai/sigmae/Control%20Tokens/runs/ho0hpvh3` |
|warmup_tinyllama_pause_detached KL on one-hoe probs|`/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-24-37/test_results.json`     | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-24-37/final` | `https://wandb.ai/sigmae/Control%20Tokens/runs/bf8fjkzp` |
|warmup tinyllama no pause (baseline)               |`/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_14-35-04/test_results.json`     | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_14-35-04/final` | `https://wandb.ai/sigmae/Control%20Tokens/runs/a3bmvlbk` |





## Stage 2: Reinforcement Learning on GSM8K

