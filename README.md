# PauseToken

## Installation

1. Create a new conda envrionment:
    ```
    conda create -n pausetok python=3.11
    conda activate pausetok
    ````
2. Installing pytorch. I haven't found a version that works for both on runai and on iccluster 🥲 (something with the `trl` library causes problems)so here's my solution:
    - **If you're on Runai**, install the following requirements for torch:
        ```
        pip install -r runai_torch_requirements.txt
        ```
    - **If you're on the iccluster**, install the following requirements for torch:
        ```
        pip install -r torch_requirements.txt
        ```
3. Install the rest of the requirements:
    ```
    pip install -r pip_requirements.txt
    ```

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

Here is an example of how to use the script with the default parameters:
```bash
cd data_generation
python gsm8k_pause_injector.py --dataset_location ../data/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{"=": 1, "\n": 1," equals ":1, " equal ": 1}' --augm_dataset_save_location ../data/gsm8k_pause_injected --pause_augm_col_name "answer" --verbose --n_random_pauses 0
```

## Train LLaMa
