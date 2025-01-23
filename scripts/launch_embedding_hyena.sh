#!/bin/bash

model_name="HyenaDNA"
model_versions=(
    "hyenadna-large-1m-seqlen-hf"
    "hyenadna-medium-450k-seqlen-hf"
    "hyenadna-medium-160k-seqlen-hf"
    "hyenadna-small-32k-seqlen-hf"
    "hyenadna-tiny-16k-seqlen-d128-hf"
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
