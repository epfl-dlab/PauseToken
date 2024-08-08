from lm_stable_baselines.environments.language_model_env import LanguageModelEnv
from lm_stable_baselines.rewards import AbstractReward
from transformers import GPT2Tokenizer



# requires the following inputs:
    # reward: AbstractReward,
    # tokenizer: PreTrainedTokenizer,
    # termination_tokens: List[int],
    # max_tokens: int,
    # dataset: Dataset = None,
    # filler_token: int = -100,



# Create a reward function
class Reward(AbstractReward):
    def compute(self, observation: str, action: str, next_observation: str) -> float:
        return 1.0
    
# Create a tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

termination_tokens = [tokenizer.eos_token_id]
max_tokens = 100
env = LanguageModelEnv(reward=Reward(), tokenizer=tokenizer, termination_tokens=termination_tokens, max_tokens=max_tokens)
env.reset()