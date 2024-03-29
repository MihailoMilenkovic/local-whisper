#!/bin/bash

# model_size="large-v3"
model_size="tiny"

# base_model="true"
base_model="false"
use_lora="true"

# Parse long command-line arguments
TEMP=$(getopt -o s:b:l: --long model-size:,base-model:,use-lora: -n 'script.sh' -- "$@")
eval set -- "$TEMP"

# Process command-line arguments
while true; do
  case "$1" in
    -s|--model-size)
      model_size="$2"
      shift 2
      ;;
    -b|--base-model)
      base_model="$2"
      shift 2
      ;;
    -l|--use-lora)
      use_lora="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Internal error!"
      exit 1
      ;;
  esac
done

# Output the values of keyword arguments
echo "model_size: $model_size"
echo "base_model: $base_model"
echo "use_lora: $use_lora"

save_path="./models/$model_size"
base_model_location=openai/whisper-$model_size
dataset_location="./datasets/common-voice-serbian-cyrilic/validation"

if [ "$base_model" = "true" ]; then
    python eval.py \
        --model_ckpt_location $base_model_location \
        --dataset_location $dataset_location
else
    if [ "$use_lora" = "true" ]; then
        lora_ckpt_location="${save_path}_lora"
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