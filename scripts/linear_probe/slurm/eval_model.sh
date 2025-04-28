#!/bin/bash
#SBATCH --job-name=linear_probe_all
#SBATCH --output=slurm_out/linear_probe_all_%A_%a.out
#SBATCH --error=slurm_out/linear_probe_all_%A_%a.err
#SBATCH --partition=gpu,morrisq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
# Adjust the --array range (0-3) to match the number of model_versions minus one.

# Define an array of model_versions
model_versions=(
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_new_splice_to_toga_mask_value2_mask_val_0.25"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_new_splice_to_toga_mask_value_mask_val_-25"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_new_splice_to_toga_combined_load_new_load_new_combined_s:toga"
)
ckpt_names=(
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
    "epoch=10-step=20000.ckpt"
)
# boolean input
mask_out_splice_tracks=(
    "False"
    "False"
    "False"
)
mask_out_cds_tracks=(
    "False"
    "False"
    "False"
)
# Select the model_version based on the SLURM_ARRAY_TASK_ID
model_version=${model_versions[$SLURM_ARRAY_TASK_ID]}
mask_out_splice_track=${mask_out_splice_tracks[$SLURM_ARRAY_TASK_ID]}
mask_out_cds_track=${mask_out_cds_tracks[$SLURM_ARRAY_TASK_ID]}
ckpt_name=${ckpt_names[$SLURM_ARRAY_TASK_ID]}

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
    --ckpt_name ${ckpt_name} \
    --mask_out_splice_track ${mask_out_splice_track} \
    --mask_out_cds_track ${mask_out_cds_track}