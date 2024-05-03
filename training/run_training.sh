#!/bin/bash


# Default values
model_size="tiny"
use_lora="true"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -s|--model-size)
        model_size="$2"
        shift 2
        ;;
        -l|--use-lora)
        use_lora="$2"
        shift 2
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Output the values of keyword arguments
echo "model_size: $model_size"
echo "use_lora: $use_lora"

model_save_folder=$(dirname "$(realpath "$0")")/models

if [ "$use_lora" = "true" ]; then
    save_path="$model_save_folder/$model_size-trained_lora"
else
    save_path="$model_save_folder/$model_size-trained_full"
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
    --save_strategy steps  \
    --save_steps 250  \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.02  \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --eval_steps 200

    # --num_train_epochs 2  \