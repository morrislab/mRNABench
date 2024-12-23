#!/bin/bash

model_name="Orthrus"
model_version="orthrus_large_6_track"
dataset_name="mrl-sugimoto"
seq_chunk_overlap=0
target_task="regression"
target_col="target"

python linear_probe.py --model_name $model_name \
    --model_version $model_version --dataset_name $dataset_name \
    --seq_chunk_overlap $seq_chunk_overlap --target_task $target_task \
    --target_col $target_col
