#!/bin/bash

#SBATCH --job-name=embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=15:00:00
#SBATCH --output=./logs/embed.%A.out
#SBATCH --error=./logs/embed.%A.err

force_recompute="False"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --d_chunk_ind) d_chunk_ind="$2"; shift ;;
        --d_num_chunks) d_num_chunks="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source [path/to/conda.sh]

# if the model_name is Helix-mRNA, activate helix_bench conda environment
if [[ "$model_name" == "Helix-mRNA" ]]; then
    conda activate [path/to/helix_bench]
else
    conda activate [path/to/mrna_bench]
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$force_recompute" == "True" ]]; then
    python ../embed_dataset.py \
        --model_name "$model_name" \
        --model_version "$model_version" \
        --dataset_name "$dataset_name" \
        --d_chunk_ind "$d_chunk_ind" \
        --d_num_chunks "$d_num_chunks" \
        --force_recompute
else
    python ../embed_dataset.py \
        --model_name "$model_name" \
        --model_version "$model_version" \
        --dataset_name "$dataset_name" \
        --d_chunk_ind "$d_chunk_ind" \
        --d_num_chunks "$d_num_chunks"
fi
