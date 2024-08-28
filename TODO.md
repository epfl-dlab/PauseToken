TODO:


- Test pipeline on mistral using STaR
    - Don't forget to set reward_threshold to -1 in the config of the buffer
- Test pipeline on pause token model

<!-- - DataCollatorForCompletionONly DONE -->

- PRETRAINING:
    - STEP 1: Train LM on GSM8K (w/out pauses) DONE
    - STEP 2: Add Pause Wrapper to LM and only train pause embedding and pause token classifier on random pause insertions DOING
    - STEP 3 (OPTIONALLY IF THINGS DON'T WORK): Train both together on GSM8K with pause token

- RL TRAINING:
    - SANITY CHECK on STAR: 
        - Look at samples taken at training
    - STaR on STEP 1 model
    - STaR on STEP 2 model !!!!! UNFREEZE ALLL

- Less URGENT:
    - Get PPO to work in this framework
        - means rewriting the policy 


