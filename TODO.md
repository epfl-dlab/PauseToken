TODO:

- Add script for doing SFT on models on a specific dataset (as a warmup for the model) DONE
    - in test function, save test predictions DONE
    - evaluation metrics DONE
- Peft Option for stable basline algorithms  (probaly can mostly be done in the config)
- Saving and Loading models:
    - Do we want to save in SB3's format?
    - Do we want to just save the LM via HF ?
    - Both ?
- Test pipelineo on mistral using STaR
    - Don't forget to set reward_threshold to -1 in the config of the buffer
- Retake a look at the pause token model and how it fits in this framework
- Test pipeline on pause token model

- Get PPO to work in this framework
    - means rewriting the policy 
