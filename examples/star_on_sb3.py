
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from datasets import Dataset, load_dataset

from lm_stable_baselines.environments import LanguageModelEnv
from lm_stable_baselines.environments.vectorized_environments import LMDummyVecEnv
from lm_stable_baselines.rewards import GSM8KCorrectnessReward
from lm_stable_baselines.buffers import LMReplayBuffer
from lm_stable_baselines.training_algorithms import STaR
from lm_stable_baselines.policies import LLMBasePolicy

def create_env(reward,tokenizer, eos_token_id, max_tok, dataset, filler_token):
    return LanguageModelEnv(reward,tokenizer, [eos_token_id], max_tok, dataset, filler_token=filler_token)


if __name__ == "__main__":
    #load gpt2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map = "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.unk_token
    #get eos token id
    eos_token_id = tokenizer.eos_token_id
    #initialize reward
    
    reward = GSM8KCorrectnessReward(tokenizer)
    input_dir = "src/data/gsm8k_json/gsm8k/"
    dataset = load_dataset('json', data_files=input_dir + 'train.json', split='train')
    dataset = dataset.rename_column("question", "input_text").rename_column("answer", "output_text")
    
    env = LMDummyVecEnv(
        [
            lambda: create_env(
                reward,tokenizer,
                eos_token_id,
                1024,
                dataset,
                filler_token=tokenizer.pad_token_id),
            lambda: create_env(
                reward,
                tokenizer,
                eos_token_id,
                1024,
                dataset,
                filler_token=tokenizer.pad_token_id
            ),
        ]
    )
    policy = LLMBasePolicy
    policy_kwargs = {"lm": model, "tokenizer": tokenizer}
    
    algo = STaR(
        policy = policy,
        policy_kwargs = policy_kwargs,
        env=env,
        learning_rate=1e-5,
        train_freq = TrainFreq(1000, TrainFrequencyUnit.STEP),
        replay_buffer_class= LMReplayBuffer,
        replay_buffer_kwargs={"tokenizer": tokenizer, "reward_threshold": reward.get_min_reward()},
        batch_size = 8,
    )
    #Note: You will get the error
    #  File "numpy/random/_bounded_integers.pyx", line 1334, in numpy.random._bounded_integers._rand_int64
    # ValueError: high <= 0
    # this will fail because gpt2 is doesn't generate good enough rewards and the "reward_threshold" argument filters on rewards
    algo.learn(100)
