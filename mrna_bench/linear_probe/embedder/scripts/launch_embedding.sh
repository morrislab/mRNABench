#!/bin/bash

max_index=10
script_name="embed.sh"
seq_overlap=511
dataset_name="go-mf"
model_name="AIDO.RNA"
model_version="aido_rna_1b600m_cds"

# Loop through indices from 1 to max_index
for i in $(seq 0 $max_index)
do
    sbatch "$script_name" $i $max_index $dataset_name $model_name $model_version $seq_overlap 
done
