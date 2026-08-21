#!/bin/bash

#SBATCH --job-name=embed
#SBATCH --partition=gpu,morrisq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=12:00:00
#SBATCH --output=./logs/embed.%A.out
#SBATCH --error=./logs/embed.%A.err

force_recompute="False"
batch_size=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --d_chunk_ind) d_chunk_ind="$2"; shift ;;
        --d_num_chunks) d_num_chunks="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source /home/dalalt1/miniforge3/etc/profile.d/conda.sh
conda activate mrnabench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$force_recompute" == "True" ]]; then
    python ../embed_dataset.py \
        --model_name "$model_name" \
        --model_version "$model_version" \
        --dataset_name "$dataset_name" \
        --d_chunk_ind "$d_chunk_ind" \
        --d_num_chunks "$d_num_chunks" \
        --batch_size "$batch_size" \
        --force_recompute
else
    python ../embed_dataset.py \
        --model_name "$model_name" \
        --model_version "$model_version" \
        --dataset_name "$dataset_name" \
        --d_chunk_ind "$d_chunk_ind" \
        --d_num_chunks "$d_num_chunks" \
        --batch_size "$batch_size"
fi
