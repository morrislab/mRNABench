#!/bin/bash

# model_name="AIDO.RNA"
# model_version="aido_rna_1b600m_cds"

# model_name="Orthrus"
# model_version="orthrus_large_6_track"

# model_name="RNA-FM"
# model_version="mrna-fm"

# model_name="DNABERT2"
# model_version="dnabert2"

model_name="NucleotideTransformer"
model_version="v2-50m-multi-species"

dataset_name="go-mf"
n_chunks=0
seq_overlap=0


if [ $n_chunks -eq 0 ]; then
    sbatch embed_slurm.sh $model_name $model_version $dataset_name 0 $n_chunks $seq_overlap
else
    # Loop through indices from 1 to max_index
    for i in $(seq 0 $((n_chunks - 1)))
    do
        sbatch embed_slurm.sh $model_name $model_version $dataset_name $i $n_chunks $seq_overlap
    done
fi
