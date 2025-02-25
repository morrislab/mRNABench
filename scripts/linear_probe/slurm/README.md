# AIDO.RNA

## Binary Essentiality

```bash
sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m_cds lncrna-ess-hap1 classification essential_HAP1 ss True
sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m_cds pcg-ess-hap1 classification essential_HAP1 ss True

sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m_cds lncrna-ess-hap1 classification essential_HAP1 ss True
sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m_cds pcg-ess-hap1 classification essential_HAP1 ss True

sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m_cds pcg-ess-shared classification essential_SHARED ss True
sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m_cds lncrna-ess-shared classification essential_SHARED ss True

sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m_cds pcg-ess-shared classification essential_SHARED ss True
sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m_cds lncrna-ess-shared classification essential_SHARED ss True
```
## Day 14 LogFc

```bash
sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m_cds lncrna-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
sbatch modelname_slurm.sh AIDO.RNA aido_rna_1b600m_cds pcg-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True

sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m_cds lncrna-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
sbatch modelname_slurm.sh AIDO.RNA aido_rna_650m_cds pcg-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
```

# RiNALMo
## Binary Essentiality

```bash 
sbatch modelname_slurm.sh RiNALMo rinalmo lncrna-ess-hap1 classification essential_HAP1 ss True
sbatch modelname_slurm.sh RiNALMo rinalmo pcg-ess-hap1 classification essential_HAP1 ss True

sbatch modelname_slurm.sh RiNALMo rinalmo pcg-ess-shared classification essential_SHARED ss True
sbatch modelname_slurm.sh RiNALMo rinalmo lncrna-ess-shared classification essential_SHARED ss True
```
## Day 14 LogFc

```bash
sbatch modelname_slurm.sh RiNALMo rinalmo lncrna-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
sbatch modelname_slurm.sh RiNALMo rinalmo pcg-ess-hap1 reg_ridge day14_log2fc_HAP1 ss True
```