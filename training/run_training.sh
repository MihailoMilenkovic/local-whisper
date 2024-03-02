#!/bin/bash
#SBATCH --job-name=train-whisper-serbian
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --partition=cuda
#SBATCH --nodelist=n17


# model_size="large-v3"
model_size="tiny"

python train.py \
    --model_path openai/whisper-$model_size \
    --train_dataset_path ./datasets/common-voice-serbian-cyrilic/train \
    --eval_dataset_path ./datasets/common-voice-serbian-cyrilic/validation \
    --use_peft true \
    --peft_mode lora \
    --lora_rank 16   \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --output_dir models/$model_size \
    --save_strategy "no"  \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.02  \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --eval_steps 100

    # --num_train_epochs 2  \