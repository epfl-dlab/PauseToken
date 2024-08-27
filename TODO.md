TODO:


- Test pipeline on mistral using STaR
    - Don't forget to set reward_threshold to -1 in the config of the buffer
- Test pipeline on pause token model


- PRETRAINING:
    - STEP 1: Train LM on GSM8K (w/out pauses)
    - STEP 2: Add Pause Wrapper to LM and only train pause embedding and pause token classifier on random pause insertions
    - STEP 3 (OPTIONALLY IF THINGS DON'T WORK): Train both together on GSM8K with pause token

- Get PPO to work in this framework
    - means rewriting the policy 


