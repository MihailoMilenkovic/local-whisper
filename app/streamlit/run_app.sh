script_dir=$(dirname "$(realpath "$0")")

model_base=openai/whisper-medium
model_save_location="medium-trained_lora-common-voice-serbian-latin"
model_ckpt=$(dirname $(dirname $(dirname $script_dir)))/training/models/$model_save_location

#NOTE: extra -- to bypass streamlit cli args and add custom cli args for script
streamlit run $script_dir/app.py \
  -- --model_ckpt_location $model_ckpt \
     --base_model_location $model_base