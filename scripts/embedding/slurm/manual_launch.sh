#!/bin/bash

model_name=Evo2
model_versions=(
    # "evo2_40b"
    # "evo2_7b"
    # "evo2_40b_base"
    # "evo2_7b_base"
    "evo2_1b_base"
)

dataset_names=(
    # "eclip-binding-k562"
    # "eclip-binding-hepg2"
    # "go-mf"
    # "go-bp"
    # "go-cc"
    # "rnahl-human"
    # "rnahl-mouse"
    # "rna-loc-fazal"
    # "rna-lifecycle-ietswaart"
    # "prot-loc"
    # "mrl-hl-lbkwk" # needs only 1 chunk
    # "mrl-sugimoto"
    # "vep-mapsy"
    # "vep-traitgym-complex"
    # "vep-traitgym-mendelian"
    # "mirna-target"
    # "mrl-sample-egfp" # needs like 25 chunks
    # "mrl-sample-mcherry" # needs like 25 chunks
    # "mrl-sample-designed" # needs like 25 chunks
    # "mrl-sample-varying" # needs like 25 chunks
    # "pal-tail-length-xiang-gv"
    # "pal-tail-length-xiang-gvtomii"
    # "pal-tail-length-xiang-p4initial"
    # "pal-tail-length-xiang-p4diff"
    # "utr-variants-bohn-utr5"
    # "utr-variants-bohn-utr3"
    # "translation-efficiency-mouse"
    # "translation-efficiency-human"
)

max_chunks=3
force_recompute="False"

for version in "${model_versions[@]}"; do

    echo "Running for model version: $version"

    for dataset_name in "${dataset_names[@]}"; do

        echo "Running for dataset: $dataset_name"

        if [[ $max_chunks -eq 1 ]]; then
            sbatch evo_slurm_script.sh \
                --model_name $model_name \
                --model_version $version \
                --dataset_name $dataset_name \
                --d_chunk_ind 0 \
                --d_num_chunks 0 \
                --force_recompute $force_recompute
        else
            for ((chunk_ind=0; chunk_ind<max_chunks; chunk_ind++)); do

                echo "${chunk_ind} / $max_chunks"

                sbatch evo_slurm_script.sh \
                    --model_name $model_name \
                    --model_version $version \
                    --dataset_name $dataset_name \
                    --d_chunk_ind $chunk_ind \
                    --d_num_chunks $max_chunks \
                    --force_recompute $force_recompute
            done
        fi
    done
done