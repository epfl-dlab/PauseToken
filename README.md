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
- `variable_number_of_pauses`: Enable variable number of pauses in sequence (w/ max being n_random_pauses, U[0, n_random_pauses] pauses per sequence)
- `n_generated_samples_per_datapoint`: The number of samples to be generated per datapoint in the dataset (number of y's to be generated per x)

Here is an example of how to use the script with the default parameters:
```bash
cd data_generation
python gsm8k_pause_injector.py --dataset_location ../data/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{"=": 1, "\n": 1," equals ":1, " equal ": 1}' --augm_dataset_save_location ../data/gsm8k_pause_injected --pause_augm_col_name "answer" --verbose --n_random_pauses 0 --tokenizer_hf_name "/dlabdata1/llm_hub//Mistral-7B-v0.1"

#DS For Pretrain
python gsm8k_pause_injector.py --dataset_location ../data/gsm8k_jsonl/gsm8k --pause_token "<|pause|>" --n_pauses_per_patterns '{}' --augm_dataset_save_location ../data/gsm8k_jsonl/gsm8k_random_pauses_5_samples_per_dp --pause_augm_col_name "answer" --verbose --n_random_pauses 100 --tokenizer_hf_name "/dlabdata1/llm_hub//Mistral-7B-v0.1" --variable_number_of_pauses --n_generated_samples_per_datapoint 5 --verbose
```
## Train Models

To fine-tune a model on the training data: 
```
python src/sft.py --model-name=mistralai/Mistral-7B-v0.1 --n-epochs=2 --batch-size=8 --logging-steps=50 --use-peft=true --data-dir=data/gsm8k_jsonl/gsm8k --task=gsm8k --max-length=512 --save-steps=500 --eval-steps=300
```

To fine-tune a model with pause token: 
```
python3 src/sft_pause.py --model-name=mistralai/Mistral-7B-v0.1 --n-epochs=2 --batch-size=8 --logging-steps=50 --use-peft=true --data-dir=data/gsm8k_jsonl/gsm8k_10_random_pause_injected_mistral --task=gsm8k --max-length=512 --save-steps=500 --eval-steps=300
```

Evaluate models performance: 
```
python3 src/eval_script.py --model-path XXXX --test-data-path XXXX --output-file-name XXX
```

Pretrain model w/ random pauses (ignore pause tokens in loss)
```
cd src
python pretrain.py --data-dir "../data/gsm8k_jsonl/" --model-name "/dlabdata1/llm_hub//Mistral-7B-v0.1" --n-epochs 3 --task "gsm8k_random_pauses_5_samples_per_dp" --batch-size 8 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag pretrain_mistral --modules-to-save embed_tokens lm_head
```

Training WSFT:

A brief description of some parameters specific to wsft:
- `n-outer-loops`: Number of outer loops = 1 round of WSFT + 1 rollout (except for the first round, where we do SFT on the original DS)
- `batch-size-rollout`: Batch size for the rollout
- `n-samps-per-prompt-rollout`: Number of samples per prompt in the rollout (i.e., the number of y's generated for a single x)
- `pause-temperature`: Temperature for the pause token (see Overleaf Pause temperature (see 5.3 of Overleaf Amortized Search For Language Model Decoding)). It attenuates (`0  \leq pause-temperature < 1`) of increases (`1 < pause-temperature`) the probability of sampling the pause token.
- `wsft-beta`: Beta parameter for WSFT
```
python wsft.py --data-dir "/dlabdata1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabdata1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout <N-SAMPS-PER-ROLLOUT> --task "gsm8k_10_random_pause_injected_mistral" --batch-size 8 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag <YOUR-TAG>--modules-to-save embed_tokens lm_head --pause-temperature 0.5 --wsft-beta <BETA-PARAMETER>
```

Behavior Cloning with invariant LM:
```
python behavior_cloning.py --data-dir "/dlabdata1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabdata1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 2 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 8 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag <YOUR-TAG> --modules-to-save embed_tokens lm_head --pause-temperature 0.5 --disable-peft
```

python testing_inv_modeling.py --data-dir "/dlabdata1/baldwin/PauseToken/data/gsm8k_jsonl" --model-name "/dlabdata1/baldwin/PauseToken/src/models/gsm8k_random_pauses_5_samples_per_dp/pretrain_mistral/pretrain_Mistral-7B-v0.1_trl_2024-05-15_10:16:32.770577" --n-epochs 1 --n-outer-loops 3 --batch-size-rollout 32 --n-samps-per-prompt-rollout 2 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 8 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --tag tst --modules-to-save embed_tokens lm_head --pause-temperature 0.5 --disable-peft


<!-- ## Train LLaMa -->


<!-- ## Reward Conditioned:
- <ins> train </ins>:
    - **With Mistral**:
        ```
        cd src
        python reward_conditioned.py --data-dir "/dlabdata1/baldwin/PauseToken/data/" --model-name "/dlabdata1/llm_hub/Mistral-7B-v0.1" --n-epochs 1 --task "gsm8k_10_random_pause_injected_mistral" --batch-size 8 --logging-steps=50 --max-length=300 --save-steps=500 --eval-steps=3000 --modules-to-save embed_tokens lm_head
        ```
- <ins> inference </ins>:
    ```
    cd src
    python run_inference_rc.py --model-path <PATH-TO-YOUR-MODEL> --test-data-path /dlabdata1/baldwin/PauseToken/data/gsm8k/test.json --output-filename <NAME-OF-YOUR-OUTPUT-FILE-NAME>
    ``` -->