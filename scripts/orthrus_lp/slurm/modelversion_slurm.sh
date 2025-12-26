#!/bin/bash
#SBATCH --job-name=orthrus_lp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=2:00:00
#SBATCH --output=./logs/orthrus_lp.%A.out
#SBATCH --error=./logs/orthrus_lp.%A.err

# set default value for force_recompute
force_recompute="False"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_short_name) model_short_name="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --task) task="$2"; shift ;;
        --target) target="$2"; shift ;;
        --split_type) split_type="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source [path/to/conda.sh]
conda activate [path/to/env]

# if force_recompute is set to True
if [ "$force_recompute" == "True" ]; then
    python ../by_modelversion.py \
    --model_short_name "$model_short_name" \
    --dataset_name "$dataset_name" \
    --task "$task" \
    --target "$target" \
    --split_type "$split_type" \
    --force_recompute
else
    python ../by_modelversion.py \
    --model_short_name "$model_short_name" \
    --dataset_name "$dataset_name" \
    --task "$task" \
    --target "$target" \
    --split_type "$split_type"
fi
