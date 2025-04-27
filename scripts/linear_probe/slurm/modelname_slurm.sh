#!/bin/bash

#SBATCH --output=./logs/linear_probe_%A.out
#SBATCH --error=./logs/linear_probe_%A.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

source /home/dalalt1/compute/miniforge3/etc/profile.d/conda.sh
conda activate /home/dalalt1/compute/miniforge3/envs/aido_bench

python ../by_modelname.py --model_name $1 --model_version $2 \
    --data $3 --task $4 --target $5 --seq_chunk_overlap 0 --split_type $6 --isoform_resolved $7