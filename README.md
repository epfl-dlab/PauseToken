# PauseToken

## Installation
```
conda create -n pausetok python=3.11
conda activate pausetok
pip install -r pip_requirements.txt
```

## Data Generation

To generate a pause token augmented dataset, you can tweet the following parameters:

- `dataset_location`: The location of the dataset to be augmented.
- `pause_token`: The pause token string to be used for augmentation.
- `n_pauses_per_patterns`:  dictionary of key value pairs where key is the pattern and value is the number of pauses to be injected after an occurence of that pattern"
- `augm_dataset_save_location`: The location where the augmented dataset will be saved.
- `pause_augm_col_name`: The name of the column where the augmented data will be saved in the dataset
- `verbose`: If set, the script will print the progress of the augmentation process.

Here is an example of how to use the script with the default parameters:
```bash
cd data_generation
python gs8k_pause_injector.py --dataset_location ../data/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{"=": 1, "\n": 1}' --augm_dataset_save_location ../data/gs8k_pause_injected --pause_augm_col_name "pause_augmented_answer"
```
