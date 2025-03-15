#!/bin/bash

#SBATCH --output=./logs/output_%A.log
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

python ../by_embname.py --embedding_fn $1 --dataset_name $2 --task $3 \
    --target_col $4 --split_type $5
