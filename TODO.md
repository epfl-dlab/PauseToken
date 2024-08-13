
- Make every component importable with configs and hydra:
    - Dataset (HF Dataset) 1 (masani) DONE
    - Trainer 
    - Environment 1 (masani) DONE
    - Rl-algorithm DONE
    - Policy  1 (Nicky) DONE
    - Evaluation pipelines
    - Logger 
    - buffers DONE

- For Trainer:
    - Figure out difference between learn of off-policy and on-policy
        - OFF-Policy:
            - Until current time step < total time steps:
                - Perform rollout
                - Store in buffer
                - Train
        - ON-Policy:
            - Actually is the same but just control collect rollout differently
    - build trainer accordingly
        - Call Learn with given amout of iterations
        - Then, save model, log, possible eval ?

- Logger:
    - Can we directly use wandb with their logger =

- Evaluation pipelines:
    - Implement evaluation pipelines for different environments