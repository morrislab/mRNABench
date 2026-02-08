"""Run fine-tuning for a model on a benchmark dataset."""

import argparse
import json

import numpy as np
import torch

import mrna_bench as mb
from mrna_bench.fine_tune import (
    FineTunePersister,
    FineTuneTrainer,
    FineTuneWrapper,
    TaskHead,
    TrainerConfig,
    create_dataloaders,
)
from mrna_bench.models import MODEL_CATALOG

default_seeds = "[0]"
default_lrs = "[1e-5, 1e-4, 1e-3]"
default_ranks = "[4, 8, 16]"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--model_version", type=str, default=None)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
parser.add_argument("--split_type", type=str, default=None)
parser.add_argument("--learning_rates", type=str, default=default_lrs)
parser.add_argument("--lora_ranks", type=str, default=default_ranks)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--seeds", type=str, default=default_seeds)
parser.add_argument("--eval_test", action="store_true")
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


def get_embedding_dim(model, dataset):
    """Probe embedding dimension using a short dummy sequence.

    Args:
        model: EmbeddingModel instance.
        dataset: BenchmarkDataset to check for CDS/splice tracks.

    Returns:
        Embedding dimension (int).
    """
    dummy_seq = "ATGATG"
    kwargs = {}
    if "cds" in dataset.data_df.columns:
        kwargs["cds"] = [np.zeros(len(dummy_seq), dtype=np.float32)]
    if "splice" in dataset.data_df.columns:
        kwargs["splice"] = [np.zeros(len(dummy_seq), dtype=np.float32)]

    with torch.no_grad():
        emb = model.embed([dummy_seq], **kwargs)
    return emb.shape[-1]


def get_output_dim(dataset, task, target_col):
    """Infer output dimension from dataset and task.

    Args:
        dataset: BenchmarkDataset instance.
        task: Task type (regression, classification, multilabel).
        target_col: Target column name.

    Returns:
        Output dimension (int).
    """
    if task == "regression":
        return 1
    elif task == "multilabel":
        sample_target = dataset.data_df[target_col].iloc[0]
        return len(sample_target) if hasattr(sample_target, "__len__") else 1
    else:
        return dataset.data_df[target_col].nunique()


def run_finetune(
    model_class,
    model_version,
    dataset,
    task,
    target_col,
    split_type,
    learning_rate,
    lora_rank,
    epochs,
    batch_size,
    accumulation_steps,
    random_seed,
    device,
    eval_test=False,
):
    """Run fine-tuning for a single (lr, rank, seed) configuration.

    Args:
        model_class: Model class from MODEL_CATALOG.
        model_version: Version of the model.
        dataset: BenchmarkDataset instance.
        task: Task type (regression, classification, multilabel).
        target_col: Target column name.
        split_type: Type of data split.
        learning_rate: Learning rate.
        lora_rank: LoRA rank.
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        accumulation_steps: Gradient accumulation steps.
        random_seed: Random seed for split.
        device: PyTorch device.
        eval_test: Whether to also evaluate on test set.

    Returns:
        Tuple of (val_metrics, test_metrics, history).
    """
    model = model_class(model_version, device)
    model.set_inference_mode()

    emb_dim = get_embedding_dim(model, dataset)
    output_dim = get_output_dim(dataset, task, target_col)

    head = TaskHead(input_dim=emb_dim, output_dim=output_dim, task_type=task)
    wrapper = FineTuneWrapper(model, head)
    wrapper.apply_lora(rank=lora_rank)

    param_info = wrapper.get_parameter_count()
    print("  Parameters: {} trainable / {} total".format(
        param_info["total_trainable"], param_info["backbone_total"]
    ))

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        target_col=target_col,
        split_type=split_type,
        random_seed=random_seed,
        batch_size=batch_size,
    )

    config = TrainerConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        gradient_accumulation_steps=accumulation_steps,
    )
    trainer = FineTuneTrainer(wrapper, config)
    trainer.fit(train_loader, val_loader)

    val_metrics = trainer.evaluate(val_loader)

    test_metrics = None
    if eval_test:
        test_metrics = trainer.evaluate(test_loader)

    return val_metrics, test_metrics, trainer.history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    model_class = MODEL_CATALOG[args.model_name]
    model_version = args.model_version or model_class.default_version
    model_short_name = model_class.get_model_short_name(model_version)

    dataset = mb.load_dataset(args.dataset_name)
    metadata = dataset.metadata

    task = args.task or metadata.task[0]
    target = args.target or metadata.target_col[0]
    split_type = args.split_type or metadata.default_split_type

    print("Model: {}".format(model_short_name))
    print("Dataset: {}".format(args.dataset_name))
    print("Task: {}".format(task))
    print("Target: {}".format(target))
    print("Split type: {}".format(split_type))

    seeds = json.loads(args.seeds)
    learning_rates = json.loads(args.learning_rates)
    lora_ranks = json.loads(args.lora_ranks)

    print("Learning rates: {}".format(learning_rates))
    print("LoRA ranks: {}".format(lora_ranks))
    print("Seeds: {}".format(seeds))
    print("Eval test: {}".format(args.eval_test))

    for lr in learning_rates:
        for lora_rank in lora_ranks:
            persister = FineTunePersister(
                dataset=dataset,
                model_short_name=model_short_name,
                task=task,
                target_col=target,
                split_type=split_type,
                learning_rate=lr,
                lora_rank=lora_rank,
            )

            for seed in seeds:
                result_path = persister._get_path(seed, ".json")
                if result_path.exists() and not args.force_recompute:
                    print("Results exist: lr={}, rank={}, seed={}".format(
                        lr, lora_rank, seed
                    ))
                    continue

                print("\nRunning: lr={}, rank={}, seed={}".format(
                    lr, lora_rank, seed
                ))

                val_metrics, test_metrics, history = run_finetune(
                    model_class=model_class,
                    model_version=model_version,
                    dataset=dataset,
                    task=task,
                    target_col=target,
                    split_type=split_type,
                    learning_rate=lr,
                    lora_rank=lora_rank,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps,
                    random_seed=seed,
                    device=device,
                    eval_test=args.eval_test,
                )

                print("Val metrics:")
                for key, value in val_metrics.items():
                    print("  {}: {:.4f}".format(key, value))

                if test_metrics is not None:
                    print("Test metrics:")
                    for key, value in test_metrics.items():
                        print("  {}: {:.4f}".format(key, value))

                persister.persist_run_results(
                    metrics={
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                    random_seed=seed,
                    history=history,
                )
