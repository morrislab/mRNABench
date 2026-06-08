#!/bin/bash

#SBATCH --job-name=finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/finetune.%A.out
#SBATCH --error=./logs/finetune.%A.err

source [path/to/conda.sh]
conda activate [path/to/env]

# set default values
force_recompute="False"
eval_test="False"
epochs=15
batch_size=1
accumulation_steps=1
seeds="[0]"
learning_rates="[1e-4]"
lr_schedule="none"
lora_ranks="[8]"
lora_alphas="[16]"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        --task) task="$2"; shift ;;
        --target) target="$2"; shift ;;
        --split_type) split_type="$2"; shift ;;
        --seeds) seeds="$2"; shift ;;
        --learning_rates) learning_rates="$2"; shift ;;
        --lr_schedule) lr_schedule="$2"; shift ;;
        --lora_ranks) lora_ranks="$2"; shift ;;
        --lora_alphas) lora_alphas="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --accumulation_steps) accumulation_steps="$2"; shift ;;
        --force_recompute) force_recompute="$2"; shift ;;
        --eval_test) eval_test="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cmd="python ../run_finetune.py \
    --model_name $model_name \
    --model_version $model_version \
    --dataset_name $dataset_name \
    --task $task \
    --target $target \
    --split_type $split_type \
    --seeds $seeds \
    --learning_rates $learning_rates \
    --lr_schedule $lr_schedule \
    --lora_ranks $lora_ranks \
    --lora_alphas $lora_alphas \
    --epochs $epochs \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps"

if [ "$force_recompute" == "True" ]; then
    cmd="$cmd --force_recompute"
fi

if [ "$eval_test" == "True" ]; then
    cmd="$cmd --eval_test"
fi

eval $cmd
