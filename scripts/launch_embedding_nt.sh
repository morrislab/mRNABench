#!/bin/bash

model_name="NucleotideTransformer"
model_versions=(
    "2.5b-multi-species"
    "2.5b-1000g"
    "500m-human-ref"
    "500m-1000g"
    "v2-50m-multi-species"
    "v2-100m-multi-species"
    "v2-250m-multi-species"
    "v2-500m-multi-species"
)

dataset_names=(
    "go-mf"
    "prot-loc"
    "rnahl-human"
    "rnahl-mouse"
    "mrl-sugimoto"
)

seq_overlaps=(0 250)

for version in "${model_versions[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for seq_overlap in "${seq_overlaps[@]}"; do
            sbatch embed_slurm.sh $model_name $version $dataset_name 0 0 $seq_overlap
        done
    done
done
