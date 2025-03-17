# LM Stable Baselines

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

## Data

You can download the the data here (it's just the MATH dataset in a special format and gsm8k): https://drive.google.com/file/d/1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340/view?usp=sharing

Once you've downloaded it, unzip it, create a data folder and put it in there. Your data folder should be here:
```
PauseToken
|
|-> data/
    |-> MATH_json/
        |-> train.json
        |-> test.json
    |-> gsm8k_jsonl/
        |-> train.json
        |-> test.json
|-> src/
    |-> model/
    ....
...
```

