# PauseToken

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


## Data Generation

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

### Fine-tuning Step

**Note**: in my directory I have already trained a model at `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-08-21_12-48-15/final`. So, you can also use this model to skip the fine-tuning step and directly go to the RL training step.

In order to start your RL training with a decent policy, you need to fine-tune your model to randomly insert your control token. Here is an example of how to fine-tune a model on GSM8K dataset with the pause token "<|pause|>":

- 1. **Augment GSM8K on random pause insertions** (see Data Generation section for more details). Here we will augment the GSM8K dataset's ground truth answers with random pauses (ranging from 0 to 100 pauses):
    ```bash
    python scripts/data_generation/gsm8k_pause_injector.py --dataset_location data/gsm8k_jsonl/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{}' --augm_dataset_save_location data/gsm8k_json/gsm8k_variable_random_pauses --pause_augm_col_name "answer" --verbose --n_random_pauses 100 --tokenizer_hf_name "/dlabscratch1/public/llm_weights/llm_hub/Mistral-7B-v0.1/" --variable_number_of_pauses --n_generated_samples_per_datapoint 1 --verbose --seed 42
    ```
- 2. **Fine-tune the model on GSM8K with the pause token "<|pause|>"**:
    ```bash 
    python src/trl_train.py experiment=trl_train/sft_pause
    ```


### RL Training Step
