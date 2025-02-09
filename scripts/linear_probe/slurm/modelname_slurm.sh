#!/bin/bash

#SBATCH --output=./logs/output_%A.log
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

python ../by_modelname.py --model_name $1 --model_version $2 \
    --data $3 --task $4 --target $5
