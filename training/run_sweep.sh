#!/bin/bash
# Make script exit if a command fails
set -Eeuo pipefail

model_sizes=("tiny" "small" "medium" "large")
use_lora=("false" "true") 
for size in "${model_sizes[@]}"; do
    ./run_eval.sh --model-size "$size" --base-model true --use-lora false
    for lora in "${use_lora[@]}"; do
        if [[ $size == "small" || $size == "medium" || $size == "large" ]]; then
            if [[ $lora == "false" ]]; then
                #NOTE: full training doesn't fit in memory for larger models
                continue
            fi
        fi
        ./run_training.sh --model-size "$size" --use-lora "$lora"
        ./run_eval.sh --model-size "$size" --use-lora "$lora"  --base-model false
    done
done