#!/bin/bash

model_name=AIDO.RNA #RiNALMo
model_versions=(
    # "aido_rna_1b600m"
    "aido_rna_1b600m_cds"
    # "aido_rna_650m"
    "aido_rna_650m_cds"
    # "rinalmo"
)

dataset_names=(
    # "lncrna-ess-hap1"
    # "pcg-ess-hap1"
    "lncrna-ess-shared"
    "pcg-ess-shared"
)

target_col="essential_SHARED" #"essential_HAP1" # "essential_SHARED"
isoform_resolved="True"
transcript_avg="False"

seq_overlaps=(0)


max_chunks=25

for version in "${model_versions[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for seq_overlap in "${seq_overlaps[@]}"; do
            if [[ $max_chunks -eq 1 ]]; then
                sbatch slurm_script.sh $model_name $version $dataset_name $isoform_resolved $target_col $transcript_avg 0 0 $seq_overlap
            else
                for ((chunk_ind=0; chunk_ind<max_chunks; chunk_ind++)); do
                    sbatch slurm_script.sh $model_name $version $dataset_name $isoform_resolved $target_col $transcript_avg $chunk_ind $max_chunks $seq_overlap

                done
            fi
        done
    done
done