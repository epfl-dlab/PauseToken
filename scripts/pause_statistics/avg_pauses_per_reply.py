import json
from src.utils.utils import make_summary_table
from tqdm import tqdm
TEST_RESULTS_PATHS = [
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-37-50/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-00/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-08/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-21/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-42-59/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-27/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-51/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-38-56/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-39-17/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-39-21/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-43-29/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-28/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-43-29/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-10/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-18/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-26/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-18_17-40-37/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_09-50-47/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-10-53/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/sft/runs/2024-10-21_10-08-12/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-03-19/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_10-30-30/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-29_09-56-35/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-03-40/test_results.json",
    "/dlabscratch1/baldwin/pause2/PauseToken/logs/train/runs/2024-10-30_15-05-49/test_results.json",
]

EXP_NAMES = [
    "sft.yaml",
    "sft_peft.yaml",
    "sft_unfr_lm_head.yaml",
    "sft_unfr_lm_head_peft",
    "sft_unfr_lm_head_embed",
    "sft_unfr_lm_head_embed_peft",
    "sft_fr_phead.yaml",
    "sft_fr_phead_peft.yaml",
    "sft_fr_phead_unfr_lm_head.yaml",
    "sft_fr_phead_unfr_lm_head_peft",
    "sft_fr_phead_unfr_lm_head_embed",
    "sft_fr_phead_unfr_lm_head_embed_peft",
    "sft_fr_pembed.yaml",
    "sft_fr_pembed_peft.yaml",
    "sft_fr_pembed_unfr_lm_head.yaml",
    "sft_fr_pembed_unfr_lm_head_peft",
    "sft_fr_pembed_unfr_lm_head_embed",
    "sft_fr_pembed_unfr_lm_head_embed_peft",
    "baseline (model w/out pause; peft) 1",
    "baseline (model w/out pause; peft) 2 epoch",
    "offline_star_exp/no_pause_peft",
    "offline_star_exp/no_pause_peft_unfr_lm_head.yaml",
    "offline_star_exp/pause.yaml",
    "reward_conditioning/no_pause_constant_rc.yaml",
    "reward_conditioning/pause_constant_rc.yaml",
    # "star pause outer loop 0",
    # "star pause outer loop 1",
    # "pause constant reward conditioning outer loop 0",
    # "pause constant reward conditioning outer loop 1",
]

def count_pauses(data, pause_token = "<|pause|>"):    
    return data["predicted_output"].count(pause_token)

if __name__ == "__main__":
    avg_pauses_per_reply = {}
    avg_pauses_per_reply_correct = {}
    avg_pauses_per_reply_incorrect = {}
    for name, filname in tqdm(zip(EXP_NAMES, TEST_RESULTS_PATHS), total=len(EXP_NAMES)):
        with open(filname, "r") as f:
            data = json.load(f)
            correctly_predicted_data = [d for d in data if d["test/accuracy"]]
            incorrectly_predicted_data = [d for d in data if not d["test/accuracy"]]
            avg_pauses_per_reply[name] = round(sum([count_pauses(d) for d in data]) / len(data),2)
            avg_pauses_per_reply_correct[name] = round(sum([count_pauses(d) for d in correctly_predicted_data]) / len(correctly_predicted_data),2)
            avg_pauses_per_reply_incorrect[name] = round(sum([count_pauses(d) for d in incorrectly_predicted_data]) / len(incorrectly_predicted_data),2)
    
    print("Average pauses per reply:")
    print(make_summary_table(avg_pauses_per_reply))
    print("Average pauses per reply (correctly predicted):")
    print(make_summary_table(avg_pauses_per_reply_correct))
    print("Average pauses per reply (incorrectly predicted):")
    print(make_summary_table(avg_pauses_per_reply_incorrect))