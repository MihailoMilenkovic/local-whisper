#!/bin/bash

# model_size="large-v3"
model_size="tiny"

# base_model="true"
base_model="false"
use_lora="true"
save_path="./models/$model_size"
base_model_location=openai/whisper-$model_size
dataset_location="./datasets/common-voice-serbian-cyrilic/validation"

if [ "$base_model" = "true" ]; then
    python eval.py \
        --model_ckpt_location $base_model_location \
        --dataset_location $dataset_location
else
    if [ "$use_lora" = "true" ]; then
        lora_ckpt_location="${save_path}_lora/checkpoint-500"
        python eval.py \
            --model_ckpt_location $base_model_location \
            --lora_ckpt_location $lora_ckpt_location \
            --dataset_location $dataset_location
    else
        lora_ckpt_location="${save_path}_full"
        python eval.py \
            --model_ckpt_location $finetuned_ckpt_location \
            --dataset_location $dataset_location
    fi
fi