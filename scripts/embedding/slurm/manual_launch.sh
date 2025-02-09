#!/bin/bash

model_name=Helix-mRNA
model_versions=(
    "helix-mrna"
)

dataset_names=(
    "mrl-sample-egfp"
)

seq_overlaps=(0)


max_chunks=25

for version in "${model_versions[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for seq_overlap in "${seq_overlaps[@]}"; do
            if [[ $max_chunks -eq 1 ]]; then
                sbatch slurm_script.sh $model_name $version $dataset_name 0 0 $seq_overlap
            else
                for ((chunk_ind=0; chunk_ind<max_chunks; chunk_ind++)); do
                    sbatch slurm_script.sh $model_name $version $dataset_name $chunk_ind $max_chunks $seq_overlap
                done
            fi
        done
    done
done

