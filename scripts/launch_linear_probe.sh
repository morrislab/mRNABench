#!/bin/bash

# model_name="Orthrus"
# model_version="orthrus_large_6_track"
model_name="AIDO.RNA"
model_version="aido_rna_1b600m_cds"
dataset_name="mrl-sugimoto"
seq_chunk_overlap=0
task="regression"
target_col="target"

python linear_probe.py --model_name $model_name \
    --model_version $model_version --dataset_name $dataset_name \
    --seq_chunk_overlap $seq_chunk_overlap --task $task \
    --target_col $target_col
