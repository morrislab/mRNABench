#!/bin/bash

#SBATCH --job-name=likelihood_vep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/likelihood_vep.%A.out
#SBATCH --error=./logs/likelihood_vep.%A.err

force_recompute="False"
normalization="sum"
score_batch_size=16

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --target) target="$2"; shift ;;
        --score_method) score_method="$2"; shift ;;
        --normalization) normalization="$2"; shift ;;
        --attn_implementation) attn_implementation="$2"; shift ;;
        --score_batch_size) score_batch_size="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source [path/to/conda.sh]
conda activate [path/to/env]

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

args=(
    --model_name "$model_name"
    --model_version "$model_version"
    --dataset_name "$dataset_name"
    --task likelihood_vep
    --target "$target"
    --score_method "$score_method"
    --normalization "$normalization"
    --score_batch_size "$score_batch_size"
)

if [[ -n "$attn_implementation" ]]; then
    args+=(--attn_implementation "$attn_implementation")
fi

if [[ "$force_recompute" == "True" ]]; then
    args+=(--force_recompute)
fi

python ../by_modelname.py "${args[@]}"
