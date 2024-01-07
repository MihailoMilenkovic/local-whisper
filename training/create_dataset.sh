#!/bin/bash
#SBATCH --job-name=create-audio-dataset
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=50G

python create_dataset.py \
  --dataset_save_location=datasets/common-voice-serbian-cyrilic \
  --use_cyrilic=True