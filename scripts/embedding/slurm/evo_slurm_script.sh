#!/bin/bash

#SBATCH --job-name=evo_embed
#SBATCH --partition=morrisq,gpu,gpushort
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=02:00:00
#SBATCH --output=./logs/evo_embed.%A.out
#SBATCH --error=./logs/evo_embed.%A.err

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_class) model_class="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --d_chunk_ind) d_chunk_ind="$2"; shift ;;
        --d_num_chunks) d_num_chunks="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source /home/dalalt1/compute/miniforge3/etc/profile.d/conda.sh
conda activate /home/dalalt1/compute/miniforge3/envs/evo_bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ../embed_dataset.py \
    --model_class "$model_class" \
    --model_version "$model_version" \
    --dataset_name "$dataset_name" \
    --d_chunk_ind "$d_chunk_ind" \
    --d_num_chunks "$d_num_chunks"