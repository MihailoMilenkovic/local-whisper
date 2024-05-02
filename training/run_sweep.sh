#!/bin/bash
# Make script exit if a command fails
set -Eeuo pipefail

model_sizes=("tiny" "small") #"medium" "large")
use_lora=("true" "false")
eval_results="eval_results.txt"
for size in "${model_sizes[@]}"; do
    ./run_eval.sh --model-size "$size" --base-model true >> $eval_results
    for lora in "${use_lora[@]}"; do
        ./run_training.sh --model-size "$size" --use-lora "$lora"
        ./run_eval.sh --model-size "$size" --use-lora "$lora"  --base-model false >> $eval_results
    done
done