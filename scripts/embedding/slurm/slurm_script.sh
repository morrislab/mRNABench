#!/bin/bash

#SBATCH --job-name=aido_embed
#SBATCH --partition=morrisq,gpu,gpushort
#SBATCH -A morrisq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalalt1@mskcc.org
#SBATCH --output=./logs/aido_embed%A.out
#SBATCH --error=./logs/aido_embed%A.err

source /home/dalalt1/compute/miniforge3/etc/profile.d/conda.sh
conda activate /home/dalalt1/compute/miniforge3/envs/aido_bench

python ../embed_dataset.py --model_class $1 --model_version $2 \
    --dataset_name $3 --isoform_resolved $4 --target_col $5 --transcript_avg $6 --d_chunk_ind $7 --d_num_chunks $8 --s_chunk_overlap $9