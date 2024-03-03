#!/bin/bash

# model_size="large-v3"
model_size="tiny"
use_lora="true"
save_path="models/$model_size"
if [ "$use_lora" = "true" ]; then
    save_path="${save_path}_lora"
else
    save_path="${save_path}_full"
fi

python train.py \
    --model_path openai/whisper-$model_size \
    --train_dataset_path ./datasets/common-voice-serbian-cyrilic/train \
    --eval_dataset_path ./datasets/common-voice-serbian-cyrilic/validation \
    --use_peft $use_lora \
    --peft_mode lora \
    --lora_rank 16   \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --output_dir $save_path \
    --save_strategy "no"  \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.02  \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --eval_steps 100

    # --num_train_epochs 2  \