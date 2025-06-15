#!/bin/bash

model_name=Evo2 #RiNALMo
model_versions=(
    # "evo2_40b"
    "evo2_7b"
    # "evo2_40b_base"
    # "evo2_7b_base"
    # "evo2_1b_base"
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
    # "rna-loc-ietswaart"
    # "prot-loc"
    # "mrl-hl-lbkwk" # needs only 1 chunk
    # "mrl-sugimoto"
    # "vep-traitgym-complex"
    # "vep-traitgym-mendelian"
    # "mrl-sample-egfp" # needs like 25
    # "mrl-sample-mcherry" # needs like 25
    # "mrl-sample-designed" # needs like 25
    # "mrl-sample-varying" # needs like 25
    "pcg-ess-hap1"
    "pcg-ess-hek293ft"
    "pcg-ess-k562"
    "pcg-ess-mda-mb-231"
    "pcg-ess-thp1"
    "pcg-ess-shared"
    "lncrna-ess-hap1"
    "lncrna-ess-hek293ft"
    "lncrna-ess-k562"
    "lncrna-ess-mda-mb-231"
    "lncrna-ess-thp1"
)

max_chunks=3

for version in "${model_versions[@]}"; do

    echo "Running for model version: $version"

    for dataset_name in "${dataset_names[@]}"; do

        echo "Running for dataset: $dataset_name"

        if [[ $max_chunks -eq 1 ]]; then
            sbatch evo_slurm_script.sh \
                --model_class $model_name \
                --model_version $version \
                --dataset_name $dataset_name \
                --d_chunk_ind 0 \
                --d_num_chunks 0
        else
            for ((chunk_ind=0; chunk_ind<max_chunks; chunk_ind++)); do

                # echo "sbatch evo_slurm_script.sh --model_class $model_name --model_version $version --dataset_name $dataset_name --d_chunk_ind $chunk_ind --d_num_chunks $max_chunks"
                echo "${chunk_ind} / $max_chunks"

                sbatch evo_slurm_script.sh \
                    --model_class $model_name \
                    --model_version $version \
                    --dataset_name $dataset_name \
                    --d_chunk_ind $chunk_ind \
                    --d_num_chunks $max_chunks
            done
        fi
    done
done