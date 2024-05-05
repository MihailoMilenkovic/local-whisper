#!/bin/bash

dataset="common-voice"
language="serbian"
use_cyrilic="true"
# use_cyrilic="false"
if [ "$use_cyrilic" = "true" ]; then
    script_suffix="cyrilic"
    dataset_save_location="datasets/common-voice-serbian-cyrilic"
    cyrilic_tag="--use_cyrilic"
else
    script_suffix="latin"
fi
dataset_name="$dataset-$language-$script_suffix"
dataset_save_location="datasets/$dataset_name"
echo "creating dataset"
echo "language:$language"
echo "dataset:$dataset"
echo "script:$script_suffix"
echo "dataset name:$dataset_name"
echo "save location:$dataset_save_location"

python create_dataset.py \
  --dataset_save_location=$dataset_save_location \
  $cyrilic_tag