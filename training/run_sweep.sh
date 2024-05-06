#!/bin/bash
# Make script exit if a command fails
set -Eeuo pipefail

# model_sizes=("tiny" "small" "medium" "large")
model_sizes=("medium" "large")
# model_sizes=("small")
use_lora=("false" "true") 
# use_lora=("false")
# use_cyrilic=("true" "false")
use_cyrilic=("false")

for size in "${model_sizes[@]}"; do
    for script in "${use_cyrilic[@]}"; do
        for lora in "${use_lora[@]}"; do
            if [[ $size == "medium" || $size == "large" ]]; then
                if [[ $lora == "false" ]]; then
                    #NOTE: full training doesn't fit in memory for larger models
                    continue
                fi
            fi
            ./run_training.sh --model-size "$size" --use-lora "$lora" --use-cyrilic $script
            ./run_eval.sh --model-size "$size" --use-lora "$lora"  --base-model false --use-cyrilic $script
        done
        ./run_eval.sh --model-size "$size" --base-model true --use-lora false --use-cyrilic $script
    done
done