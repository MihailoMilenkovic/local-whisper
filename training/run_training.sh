#!/bin/bash


# Default values
model_size="large"
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
        -c|--use-cyrilic)
        use_cyrilic="$2"
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

effective_batch_size=16

case $model_size in
  "tiny") per_device_batch_size=16 ;;
  "small") per_device_batch_size=1 ;;
  "medium") per_device_batch_size=1 ;;
  "large") per_device_batch_size=1 ;;
  *) echo "Invalid model size: $model_size"; exit 1 ;;
esac

gradient_accumulation_steps=$(( effective_batch_size / per_device_batch_size ))
echo "Effective batch size: $effective_batch_size"
echo "Model size: $model_size"
echo "Gradient accumulation steps: $gradient_accumulation_steps"
echo "Per device batch size: $per_device_batch_size"

dataset="common-voice"
language="serbian"
if [ "$use_cyrilic" = "true" ]; then
    script_suffix="cyrilic"
    cyrilic_tag="--use_cyrilic"
else
    script_suffix="latin"
fi
dataset_name="$dataset-$language-$script_suffix"

if [ "$use_lora" = "true" ]; then
    save_path="$model_save_folder/$model_size-trained_lora-$dataset_name"
else
    save_path="$model_save_folder/$model_size-trained_full-$dataset_name"
fi

echo "language:$language"
echo "dataset:$dataset"
echo "script:$script_suffix"
echo "dataset name:$dataset_name"
echo "model save path:$save_path"

python train.py \
    --model_path openai/whisper-$model_size \
    --train_dataset_path ./datasets/$dataset_name/train \
    --eval_dataset_path ./datasets/$dataset_name/validation \
    --use_peft $use_lora \
    --peft_mode lora \
    --lora_rank 16   \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --output_dir $save_path \
    --save_strategy steps  \
    --save_steps 500  \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 3  \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size $per_device_batch_size \
    --eval_steps 500

    # --training_quantization_num_bits 8 \
    # --num_train_epochs 2  \