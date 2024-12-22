#!/bin/bash

max_index=0
script_name="embed.sh"
seq_overlap=0
dataset_name="go-mf"
model_name="Orthrus"
model_version="orthrus_large_6_track"

# Loop through indices from 1 to max_index
for i in $(seq 0 $max_index)
do
    sbatch "$script_name" $i $max_index $dataset_name $model_name $model_version $seq_overlap 
done
