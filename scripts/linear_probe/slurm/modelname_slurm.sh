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


# # Binary Essentiality

# sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m lncrna-ess-hap1 classification essential_HAP1 ss True
# sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m pcg-ess-hap1 classification essential_HAP1 ss True

# sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m lncrna-ess-hap1 classification essential_HAP1 ss True
# sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m pcg-ess-hap1 classification essential_HAP1 ss True

# sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m pcg-ess-shared classification essential_SHARED ss True
# sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m lncrna-ess-shared classification essential_SHARED ss True

# sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m pcg-ess-shared classification essential_SHARED ss True
# sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m lncrna-ess-shared classification essential_SHARED ss True

# # Day 14 LogFc

# sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m lncrna-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
# sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m pcg-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True

# sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m lncrna-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
# sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m pcg-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True