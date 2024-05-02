script_dir=$(dirname "$(realpath "$0")")

model_ckpt=openai/whisper-tiny
# model_save_location=""
# model_ckpt=$(dirname $(dirname $(dirname $script_dir)))/models/$model_save_location

#NOTE: extra -- to bypass streamlit cli args and add custom cli args for script
streamlit run $script_dir/app.py \
  -- --model_ckpt_location $model_ckpt