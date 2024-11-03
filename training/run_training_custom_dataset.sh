#!/bin/bash

# Default values
model_size="tiny"
use_lora="true"
lora_rank=128
dataset_path="./datasets"  # Default dataset path
eval="true"

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
        -d|--dataset-path)
        dataset_path="$2"
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
echo "dataset_path: $dataset_path"

model_save_folder=$(dirname "$(realpath "$0")")/models

effective_batch_size=1

case $model_size in
  "tiny") per_device_batch_size=1 ;;
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

dataset_name="music-transcription"

if [ "$use_lora" = "true" ]; then
    save_path="$model_save_folder/$model_size-trained_lora-$dataset_name"
else
    save_path="$model_save_folder/$model_size-trained_full-$dataset_name"
fi

# Update the dataset paths to use the provided dataset location
train_dataset_path="$dataset_path/train"
eval_dataset_path="$dataset_path/validation"

# Print final information
echo "dataset name: $dataset_name"
echo "model save path: $save_path"
echo "train dataset path: $train_dataset_path"
echo "eval dataset path: $eval_dataset_path"

# Run the training command with updated paths
python train.py \
    --model_path openai/whisper-$model_size \
    --train_dataset_path $train_dataset_path \
    --eval_dataset_path $eval_dataset_path \
    --use_peft $use_lora \
    --peft_mode lora \
    --lora_rank $lora_rank   \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --output_dir $save_path \
    --save_strategy steps  \
    --save_steps 500  \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 50  \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size $per_device_batch_size \
    --eval_steps 20 \
    --max_seq_len 4096

    # --training_quantization_num_bits 8 \
    # --num_train_epochs 2  \

trained_model_path=$save_path
base_model_location=openai/whisper-$model_size
eval_save_path=./eval_results/$model_size-trained_lora-$dataset_name.json

#TODO: use eval instead of train, currently testing by overfitting
if [ "$eval" = "true" ]; then
    python eval.py \
    --model_ckpt_location $trained_model_path \
    --dataset_location $train_dataset_path \
    --eval_save_path $eval_save_path \
    --base_model_location $base_model_location
fi