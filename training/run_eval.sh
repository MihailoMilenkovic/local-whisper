#!/bin/bash
#SBATCH --job-name=train-whisper-serbian
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --nodelist=n16
#SBATCH --partition=cuda

# model_size="large-v3"
model_size="tiny"

python eval.py \
    --model_path openai/whisper-$model_size \
    --dataset_path ./data \
    --lora_path models/$model_size \