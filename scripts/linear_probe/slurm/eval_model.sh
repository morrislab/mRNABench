#!/bin/bash
#SBATCH --job-name=linear_probe_all
#SBATCH --output=slurm_out/linear_probe_all_%A_%a.out
#SBATCH --error=slurm_out/linear_probe_all_%A_%a.err
#SBATCH --partition=gpu,morrisq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=6:00:00
#SBATCH --array=5-7
# Adjust the --array range (0-3) to match the number of model_versions minus one.

# Define an array of model_versions
model_versions=(
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.15_splice_all_basic_eutheria_siglip_norm_exp_w=0_bs_150"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_default_m30"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_default_m30"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_t4_default_m30_t6"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_t4_default_m30_t6"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_t6_default_m30_t6"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_t6_default_m30_t6"
    "ssm_6t_6_512_lr0.0005_wd1e-05_mask0.3_splice_only_eutheria_0_t6_default_m30_t6"
)
# boolean input
mask_out_splice_tracks=(
    "False"
    "False"
    "False"
    "True"
    "False"
    "True"
    "False"
    "True"
)
mask_out_cds_tracks=(
    "False"
    "False"
    "False"
    "False"
    "False"
    "False"
    "False"
    "True"
)
# Select the model_version based on the SLURM_ARRAY_TASK_ID
model_version=${model_versions[$SLURM_ARRAY_TASK_ID]}
mask_out_splice_track=${mask_out_splice_tracks[$SLURM_ARRAY_TASK_ID]}
mask_out_cds_track=${mask_out_cds_tracks[$SLURM_ARRAY_TASK_ID]}

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
    --ckpt_name epoch=22-step=20000.ckpt \
    --mask_out_splice_track ${mask_out_splice_track} \
    --mask_out_cds_track ${mask_out_cds_track}