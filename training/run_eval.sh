#!/bin/bash

# model_size="large-v3"
model_size="tiny"

# base_model="true"
base_model="false"
use_lora="true"
use_cyrilic="true"

model_save_folder=$(dirname "$(realpath "$0")")/models
# Parse long command-line arguments
TEMP=$(getopt -o s:b:l:c: --long model-size:,base-model:,use-lora:,use-cyrilic: -n 'script.sh' -- "$@")
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
    -c|--use-cyrilic)
      use_cyrilic="$2"
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
echo "Running eval:"
echo "model_size: $model_size"
echo "base_model: $base_model"
echo "use_lora: $use_lora"

save_path="$(dirname "$(realpath "$0")")/models"
base_model_location=openai/whisper-$model_size

script_dir=$(dirname "$(realpath "$0")")
eval_res_dir="$script_dir/eval_results"
# Check if the eval_results directory exists
if [ ! -d $eval_res_dir ]; then
  # Create the directory if it doesn't exist
  mkdir "$eval_res_dir"
fi

dataset="common-voice"
language="serbian"
if [ "$use_cyrilic" = "true" ]; then
    script_suffix="cyrilic"
    cyrilic_tag="--use_cyrilic"
else
    script_suffix="latin"
fi
dataset_location="./datasets/common-voice-serbian-$script_suffix/validation"
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

#NOTE: hack to only use device 1 to not get cuda errors
export CUDA_VISIBLE_DEVICES=0

if [ "$base_model" = "true" ]; then
    eval_save_path=$eval_res_dir/$model_size-base-$dataset_name.json
    model_ckpt_location=$base_model_location
else
    if [ "$use_lora" = "true" ]; then
        finetuned_model_ckpt_location="$model_save_folder/$model_size-trained_lora-$dataset_name"
        eval_save_path=$eval_res_dir/$model_size-trained_lora-$dataset_name.json
    else
        finetuned_model_ckpt_location="$model_save_folder/$model_size-trained_full-$dataset_name"
        eval_save_path=$eval_res_dir/$model_size-trained_full-$dataset_name.json
    fi
    model_ckpt_location=$finetuned_model_ckpt_location
        # --lora_ckpt_location $lora_ckpt_location \
fi
echo "Eval save path:$eval_save_path"
echo "Model ckpt location:$model_ckpt_location"

python eval.py \
    --model_ckpt_location $model_ckpt_location \
    --dataset_location $dataset_location \
    --eval_save_path $eval_save_path \
    --base_model_location $base_model_location