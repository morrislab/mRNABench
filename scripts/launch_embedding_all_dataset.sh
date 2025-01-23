#!/bin/bash

model_name="Orthrus"
model_versions=(
    "orthrus-base-4-track"
)

dataset_names=(
    "go-mf"
    "prot-loc"
    "rnahl-human"
    "rnahl-mouse"
    "mrl-sugimoto"
)

seq_overlaps=(0)

for version in "${model_versions[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for seq_overlap in "${seq_overlaps[@]}"; do
            sbatch embed_slurm.sh $model_name $version $dataset_name 0 0 $seq_overlap
        done
    done
done
