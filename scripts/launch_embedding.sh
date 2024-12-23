#!/bin/bash

# model_name="AIDO.RNA"
# model_version="aido_rna_1b600m_cds"
model_name="Orthrus"
model_version="orthrus_large_6_track"
dataset_name="mrl-sugimoto"
max_index=0
seq_overlap=0

# Loop through indices from 1 to max_index
for i in $(seq 0 $max_index)
do
    sbatch embed_slurm.sh $model_name $model_version $dataset_name $i $max_index $seq_overlap 
done
