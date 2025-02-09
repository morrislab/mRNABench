#!/bin/bash

#SBATCH --output=./logs/output_%A.log
#SBATCH --partition=morrisq,gpu
#SBATCH -A morrisq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

python ../embed_dataset.py --model_class $1 --model_version $2 \
    --dataset_name $3 --d_chunk_ind $4 --d_num_chunks $5 --s_chunk_overlap $6
