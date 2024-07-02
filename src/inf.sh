#!/bin/bash

# List of model paths
MODEL_PATHS=(
    # "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs/ilm__trl_2024-06-27_12:53:21.984677_sft"
    # "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs/ilm__trl_2024-06-27_12:53:21.984677_outer_loop_0"
    # "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs/ilm__trl_2024-06-27_12:53:21.984677_outer_loop_1"
    # "/dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs/ilm__trl_2024-06-27_12:53:21.984677_outer_loop_2"
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_no_pause/ilm__trl_2024-06-27_17:51:52.315857_sft
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_no_pause/ilm__trl_2024-06-27_17:51:52.315857_outer_loop_0
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_no_pause/ilm__trl_2024-06-27_17:51:52.315857_outer_loop_1
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_no_pause/ilm__trl_2024-06-27_17:51:52.315857_outer_loop_2
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/rs_freeze_lm/ilm__trl_2024-06-27_12:51:26.944489_outer_loop_0
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/rs_freeze_lm/ilm__trl_2024-06-27_12:51:26.944489_outer_loop_1
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/rs_freeze_lm/ilm__trl_2024-06-27_12:51:26.944489_outer_loop_2
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/rs_freeze_lm/ilm__trl_2024-06-27_12:51:26.944489_sft
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_reject_no_pause/ilm__trl_2024-06-27_23:05:13.538991_sft
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_reject_no_pause/ilm__trl_2024-06-27_23:05:13.538991_outer_loop_0
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_reject_no_pause/ilm__trl_2024-06-27_23:05:13.538991_outer_loop_1
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_reject_no_pause/ilm__trl_2024-06-27_23:05:13.538991_outer_loop_2
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_all_pairs/ilm__trl_2024-06-28_04:39:06.370696_sft
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_all_pairs/ilm__trl_2024-06-28_04:39:06.370696_outer_loop_0
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_all_pairs/ilm__trl_2024-06-28_04:39:06.370696_outer_loop_1
    # /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_pause_head_all_pairs/ilm__trl_2024-06-28_04:39:06.370696_outer_loop_2
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_best_vs_worst/ilm__trl_2024-06-28_13:44:43.582223_sft
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_best_vs_worst/ilm__trl_2024-06-28_13:44:43.582223_outer_loop_0
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_best_vs_worst/ilm__trl_2024-06-28_13:44:43.582223_outer_loop_1
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_reject_best_vs_worst/ilm__trl_2024-06-28_13:44:43.582223_outer_loop_2
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs_till_no_pauses/ilm__trl_2024-06-28_18:44:59.351298_sft
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs_till_no_pauses/ilm__trl_2024-06-28_18:44:59.351298_outer_loop_0
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs_till_no_pauses/ilm__trl_2024-06-28_18:44:59.351298_outer_loop_1
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_freeze_lm_all_pairs_till_no_pauses/ilm__trl_2024-06-28_18:44:59.351298_outer_loop_2
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_reject_no_pause/ilm__trl_2024-06-29_00:15:21.915806_sft
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_reject_no_pause/ilm__trl_2024-06-29_00:15:21.915806_outer_loop_0
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_reject_no_pause/ilm__trl_2024-06-29_00:15:21.915806_outer_loop_1
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_reject_no_pause/ilm__trl_2024-06-29_00:15:21.915806_outer_loop_2
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_all_pairs/ilm__trl_2024-06-29_05:39:00.380534_sft
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_all_pairs/ilm__trl_2024-06-29_05:39:00.380534_outer_loop_0
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_all_pairs/ilm__trl_2024-06-29_05:39:00.380534_outer_loop_1
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_all_pairs/ilm__trl_2024-06-29_05:39:00.380534_outer_loop_2
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_best_vs_worst/ilm__trl_2024-06-29_12:56:44.926955_sft
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_best_vs_worst/ilm__trl_2024-06-29_12:56:44.926955_outer_loop_0
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_best_vs_worst/ilm__trl_2024-06-29_12:56:44.926955_outer_loop_1
    /dlabscratch1/baldwin/PauseToken/src/models/gsm8k_10_random_pause_injected_mistral/dpo_ilm_best_vs_worst/ilm__trl_2024-06-29_12:56:44.926955_outer_loop_2
    # Add more model paths as needed
)

# Function to generate the output filename from the model path
generate_output_filename() {
    model_path="$1"
    parent_folder_name=$(basename $(dirname "$model_path"))
    model_name=$(basename "$model_path")

    # Remove the timestamp part
    model_suffix=$(echo "$model_name" | sed -E 's/ilm__trl_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+_//')
    
    output_filename="${parent_folder_name}_${model_suffix}.json"
    echo "$output_filename"
}
# Common parameters
TEST_DATA_PATH="/dlabscratch1/baldwin/PauseToken/data/gsm8k_json/gsm8k/test.json"
TRAIN_METHOD="wsft"
BATCH_SIZE=16

# Loop through each model path and generate the corresponding output filename
for model_path in "${MODEL_PATHS[@]}"; do
    output_filename=$(generate_output_filename "$model_path")
    
    echo "Running inference for model: $model_path"
    echo "Output will be saved to: $output_filename"
    
    python run_inference.py \
        --model-path "$model_path" \
        --test-data-path "$TEST_DATA_PATH" \
        --output-filename "$output_filename" \
        --train-method "$TRAIN_METHOD" \
        --batch-size "$BATCH_SIZE"
    
    echo "Completed inference for model: $model_path"
    echo
done

echo "All experiments completed."