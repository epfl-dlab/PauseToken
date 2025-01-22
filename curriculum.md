# Curriculum Experiments

## Running the baselines

Our baselines are pure SFT on the dataset and pure RL (no reasoning steps provided) with some warmup rounds of sft. To run the baselines run the following commands:
```shell
python src/train.py experiment=train/sft/llama1B #full sft on on LLama 1B for 10 epochs
python src/train.py experiment=train/sft/mistral #full sft on on LLama 8B for 10 epochs

#LLAMA 1B

## PPO
python src/train.py experiment=train/ppo/llama1B/baseline_sft_1_epoch # Llama1B ppo with 1 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/ppo/llama1B/baseline_sft_2_epoch # Llama1B ppo with 2 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/ppo/llama1B/baseline_sft_3_epoch # Llama1B ppo with 3 epoch of sft as warmup for 10 epochs

##STaR
python src/train.py experiment=train/online_star/llama1B/baseline_sft_1_epoch # Llama1B star with 1 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/online_star/llama1B/baseline_sft_2_epoch # Llama1B star with 2 epoch of sft as warmup for 10 epochs 
python src/train.py experiment=train/online_star/llama1B/baseline_sft_3_epoch # Llama1B star with 3 epoch of sft as warmup for 10 epochs

# Reinforce
python src/train.py experiment=train/reinforce/llama1B/baseline_sft_1_epoch  # Llama1B reinforce with 1 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/reinforce/llama1B/baseline_sft_2_epoch # Llama1B reinforce with 2 epoch of sft as warmup for 10 epochs 
python src/train.py experiment=train/reinforce/llama1B/baseline_sft_3_epoch # Llama1B reinforce with 3 epoch of sft as warmup for 10 epochs


#Mistral

## PPO
python src/train.py experiment=train/ppo/mistral/baseline_sft_1_epoch # mistral ppo with 1 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/ppo/mistral/baseline_sft_2_epoch # mistral ppo with 2 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/ppo/mistral/baseline_sft_3_epoch # mistral ppo with 3 epoch of sft as warmup for 10 epochs

##STaR
python src/train.py experiment=train/online_star/mistral/baseline_sft_1_epoch # mistral star with 1 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/online_star/mistral/baseline_sft_2_epoch # mistral star with 2 epoch of sft as warmup for 10 epochs 
python src/train.py experiment=train/online_star/mistral/baseline_sft_3_epoch # mistral star with 3 epoch of sft as warmup for 10 epochs

# Reinforce
python src/train.py experiment=train/reinforce/mistral/baseline_sft_1_epoch  # mistral reinforce with 1 epoch of sft as warmup for 10 epochs
python src/train.py experiment=train/reinforce/mistral/baseline_sft_2_epoch # mistral reinforce with 2 epoch of sft as warmup for 10 epochs 
python src/train.py experiment=train/reinforce/mistral/baseline_sft_3_epoch # mistral reinforce with 3 epoch of sft as warmup for 10 epochs

``` 

<!-- More information on the runs:

| Run Description                                          | Wandb Link                                                          |  Best Validation Accuracy    |
|----------------------------------------------------------|:-------------------------------------------------------------------:||:---------------------------:|
| full sft on on LLama 1B for 10 epochs                    | [Click Here](https://wandb.ai/epfl-dlab/SFT/runs/f8abpycy)          |                              |
| full sft on on LLama 8B for 10 epochs                    | [Click Here](https://wandb.ai/epfl-dlab/SFT/runs/x0boe12l)          |                              |
| Llama8B ppo with 1 epoch of sft as warmup for 10 epochs  | [Click Here](https://wandb.ai/epfl-dlab/ppo-on-gsm8k/runs/wgit1uty) |                              |
| Llama8B ppo with 2 epoch of sft as warmup for 10 epochs  | [Click Here](https://wandb.ai/epfl-dlab/ppo-on-gsm8k/runs/24pw4487) |                              |
| Llama8B ppo with 3 epoch of sft as warmup for 10 epochs  | [Click Here](https://wandb.ai/epfl-dlab/ppo-on-gsm8k/runs/ktbwreyu) |                              |
| Llama1B ppo with 1 epoch of sft as warmup for 10 epochs  | [Click Here](https://wandb.ai/epfl-dlab/ppo-on-gsm8k/runs/ca8lhpbu) |                              |
| Llama1B ppo with 2 epoch of sft as warmup for 10 epochs  | [Click Here](https://wandb.ai/epfl-dlab/ppo-on-gsm8k/runs/1mc7sw7z) |                              |
| Llama1B ppo with 3 epoch of sft as warmup for 10 epochs  | [Click Here](https://wandb.ai/epfl-dlab/ppo-on-gsm8k/runs/02p2zcdg) |                              | -->