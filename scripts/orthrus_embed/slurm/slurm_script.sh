#!/bin/bash
#SBATCH --job-name=orthrus_embed
#SBATCH --partition=morrisq,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=./logs/orthrus_embed.%A.out
#SBATCH --error=./logs/orthrus_embed.%A.err

# set default value for force_recompute
force_recompute="False"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_dir) model_dir="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --checkpoint) checkpoint="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source /home/dalalt1/compute/miniforge3/etc/profile.d/conda.sh
conda activate /home/dalalt1/compute/miniforge3/envs/cerberus

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# if force_recompute is set to True

if [ "$force_recompute" == "True" ]; then
    python ../embed_dataset.py \
    --model_dir "$model_dir" \
    --model_version "$model_version" \
    --checkpoint "$checkpoint" \
    --dataset_name "$dataset_name" \
    --force_recompute
else
    python ../embed_dataset.py \
    --model_dir "$model_dir" \
    --model_version "$model_version" \
    --checkpoint "$checkpoint" \
    --dataset_name "$dataset_name"
fi

