# model_size="large-v3"
model_size="tiny"

python finetune_lora.py \
    --model_path openai/whisper-$model_size \
    --dataset_path ./data \
    --lora_path models/$model_size \