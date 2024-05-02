#!/bin/bash
use_gpu="$1"

if [[ $use_gpu == "True" ]]
then
  echo "Using gpus"
  srun --pty -w n17 -p cuda --gres=gpu:2 bash
else
  echo "Not using gpus"
  srun --pty -w n17 -p cuda bash
fi
