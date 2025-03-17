# Models and their respective paths

| Name              | Description                           | Accuracy            | Path                                                                               |
|-------------------|:-------------------------------------:|:-------------------:|:----------------------------------------------------------------------------------:|
| Mistral No Pause  | Mistral Warmed up on GSM8K             |      0.532          | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-08-12/final`  |
| Mistral Pause     | Mistral Pause Model Warmed up on GSM8K |      0.327          | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-23_09-48-40/final`  |
| tinyllama No Pause| Tinyllama Warmed up on GSM8K           |      0.053          | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_14-35-04/final`  |
| tinyllama Pause   | Tinyllama Warmed up on GSM8K           |      0.058          | `/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-47-16/final`  |
| LLama3.2 1B       | Model as is                            |                     | `/dlabscratch1/public/llm_weights/meta-llama_Llama-3.2-1B`                         |
| LLama3.2 3B       | Model as is                            |                     | `/dlabscratch1/public/llm_weights/meta-llama_Llama-3.2-3b`                         |
| LLama2-7B         | Model as is                            |                     | `/dlabscratch1/public/llm_weights/llama2_hf/Llama-2-7b-hf`                         |
| LLama3.1-8B       | Model as is                            |                     | `/dlabscratch1/public/llm_weights/llama3.1_hf/Meta-Llama-3.1-8B`                   |
