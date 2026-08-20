#!/bin/bash

#SBATCH --job-name=linear_probe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=5:00:00
#SBATCH --output=./logs/linear_probe.%A.out
#SBATCH --error=./logs/linear_probe.%A.err

source [path/to/conda.sh]
conda activate [path/to/env]

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# set default value for force_recompute
force_recompute="False"
combo=""
regressor="ols"
seeds="[2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --task) task="$2"; shift ;;
        --regressor) regressor="$2"; shift ;;
        --target) target="$2"; shift ;;
        --split_type) split_type="$2"; shift ;;
        --combo) combo="$2"; shift ;;
        --seeds) seeds="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# if force_recompute is set to True
if [ "$force_recompute" == "True" ]; then
    python ../by_modelname.py \
    --model_name "$model_name" \
    --model_version "$model_version" \
    --dataset_name "$dataset_name" \
    --task "$task" \
    --regressor "$regressor" \
    --target "$target" \
    --split_type "$split_type" \
    --combo "$combo" \
    --seeds "$seeds" \
    --force_recompute
else
    python ../by_modelname.py \
    --model_name "$model_name" \
    --model_version "$model_version" \
    --dataset_name "$dataset_name" \
    --task "$task" \
    --regressor "$regressor" \
    --target "$target" \
    --split_type "$split_type" \
    --combo "$combo" \
    --seeds "$seeds"
fi
