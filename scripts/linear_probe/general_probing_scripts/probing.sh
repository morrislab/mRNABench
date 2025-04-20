#!/bin/bash
#SBATCH --job-name=orthrus_probe
#SBATCH --partition=morrisq,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=2:30:00
#SBATCH --output=./logs/orthrus_probe_%j.out
#SBATCH --error=./logs/orthrus_probe_%j.err

# Activate the conda environment.
# conda activate [insert env]

# Run the Python script for the specified model.
python run_probing.py --model-dir $1 --results-dir $2 --default-naming