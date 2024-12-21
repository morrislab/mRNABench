#!/bin/bash

#SBATCH --output=logs/output_%A.log
#SBATCH --partition=morrisq
#SBATCH -A morrisq
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

cd ..
python embed_dataset.py --model_class $4 --model_version $5 \
    --dataset $3 --embedding_dir /data1/morrisq/ian/lp_embeddings \
    --d_chunk_ind $1 --d_chunk_max_ind $2 --s_chunk_overlap $6
