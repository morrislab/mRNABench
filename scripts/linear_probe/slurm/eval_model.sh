#!/bin/bash
#SBATCH --job-name=linear_probe_all
#SBATCH --output=slurm_out/linear_probe_all_%A_%a.out
#SBATCH --error=slurm_out/linear_probe_all_%A_%a.err
#SBATCH --partition=gpu,morrisq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
# Adjust the --array range (0-3) to match the number of model_versions minus one.

# Define an array of model_versions
model_versions=(
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_underweight_ortho_self_0.8_05_95"
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_underweight_ortho_self_0.5_05_95"
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_dual_heads_05_95"
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_published_95_05"
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_published_50_50"
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_published_05_95"
    "ssm_6t_6_512_lr0.001_wd5e-05_mask0.15_new_splice_to_toga_combined_15_mask_mlm_1.0_underweight_ortho_self_05_95"
)
# boolean input
ckpt_name=(
    "epoch=6-step=20000.ckpt"
    "epoch=6-step=20000.ckpt"
    "epoch=6-step=20000.ckpt"
    "epoch=6-step=20000.ckpt"
    "epoch=6-step=20000.ckpt"
    "epoch=6-step=20000.ckpt"
    "epoch=6-step=20000.ckpt"
)
# Select the model_version based on the SLURM_ARRAY_TASK_ID
model_version=${model_versions[$SLURM_ARRAY_TASK_ID]}
ckpt_name=${ckpt_name[$SLURM_ARRAY_TASK_ID]}
echo "Selected model_version: $model_version"

# prepare your environment here
echo date: Job $SLURM_JOB_ID is allocated resource
echo "Starting task $SLURM_ARRAY_TASK_ID"

# virtual env
eval "$(conda shell.bash hook)"
conda activate mrna_bench

cd /home/fradkinp/Documents/01_projects/mRNABench/scripts/linear_probe

python linear_probe_all.py --model_name Orthrus \
    --model_version ${model_version} \
    --ckpt_name ${ckpt_name} 