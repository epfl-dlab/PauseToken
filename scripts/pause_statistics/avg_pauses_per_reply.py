import json
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.utils import make_summary_table
from tqdm import tqdm
from transformers import AutoTokenizer

TEST_RESULTS_PATHS = [
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-47-16/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-17_10-45-38/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-24-57/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-24-37/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_14-35-04/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-23_09-48-40/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-23_09-48-40/test_results.json"
]

EXP_NAMES = [
    "warmup_tinyllama_pause_detached KL on true probs",   
    "warmup_tinyllama_pause KL on true probs",            
    "warmup_tinyllama_pause KL on one-hot probs",        
    "warmup_tinyllama_pause_detached KL on one-hot probs",
    "Warmup tinyllama no pause (baseline)", 
    "tdt",
    "warmup mistral"
]

def count_pauses(data, pause_token = "<|pause|>"):    
    return data["predicted_output"].count(pause_token)

def percent_pause_tokens(data, tokenizer, pause_token = "<|pause|>"):
    tokenized_prediction = tokenizer(data["predicted_output"])["input_ids"]
    n_pauses = data["predicted_output"].count(pause_token)
    return float(n_pauses) / len(tokenized_prediction)

if __name__ == "__main__":
    avg_pauses_per_reply = {}
    avg_pauses_per_reply_correct = {}
    avg_pauses_per_reply_incorrect = {}
    percent_of_pauses = {}
    tokenizer = AutoTokenizer.from_pretrained("/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-12-16_17-20-00/final/")
    for name, filname in tqdm(zip(EXP_NAMES, TEST_RESULTS_PATHS), total=len(EXP_NAMES)):
        with open(filname, "r") as f:
            data = json.load(f)
            correctly_predicted_data = [d for d in data if d["test/accuracy"]]
            incorrectly_predicted_data = [d for d in data if not d["test/accuracy"]]
            avg_pauses_per_reply[name] = round(sum([count_pauses(d) for d in data]) / len(data),2)
            avg_pauses_per_reply_correct[name] = round(sum([count_pauses(d) for d in correctly_predicted_data]) / len(correctly_predicted_data),2)
            avg_pauses_per_reply_incorrect[name] = round(sum([count_pauses(d) for d in incorrectly_predicted_data]) / len(incorrectly_predicted_data),2)
            percent_of_pauses[name] = round(sum([percent_pause_tokens(d, tokenizer) for d in data]) / len(data),2)
    print("Average pauses per reply:")
    print(make_summary_table(avg_pauses_per_reply))
    print("Average pauses per reply (correctly predicted):")
    print(make_summary_table(avg_pauses_per_reply_correct))
    print("Average pauses per reply (incorrectly predicted):")
    print(make_summary_table(avg_pauses_per_reply_incorrect))
    print("Percent of tokens that are pause tokens:")
    print(make_summary_table(percent_of_pauses))