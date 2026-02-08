"""Run fine-tuning for a model on a benchmark dataset."""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import mrna_bench as mb
from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.fine_tune import (
    FineTunePersister,
    FineTuneTrainer,
    SequenceDataset,
    TaskHead,
    TrainerConfig,
    make_fine_tunable,
)
from mrna_bench.models import MODEL_CATALOG

default_seeds = "[0]"
default_lrs = "[1e-4]"
default_ranks = "[8]"

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


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch keeping variable-length arrays as lists."""
    sequences = [item["sequence"] for item in batch]
    targets = np.array([item["target"] for item in batch])

    result = {
        "sequence": sequences,
        "target": targets,
    }

    if "cds" in batch[0]:
        result["cds"] = [item["cds"] for item in batch]
    if "splice" in batch[0]:
        result["splice"] = [item["splice"] for item in batch]

    return result


def create_dataloaders(
    dataset: BenchmarkDataset,
    target_col: str,
    split_type: str,
    random_seed: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from dataset splits.

    Args:
        dataset: Benchmark dataset.
        target_col: Target column name.
        split_type: Type of data split.
        random_seed: Random seed for split.
        batch_size: Batch size for dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    splits = dataset.get_splits(
        split_ratios=(0.7, 0.15, 0.15),
        random_seed=random_seed,
        split_type=split_type,
    )

    def df_to_loader(df, shuffle: bool) -> DataLoader:
        sequences = df["sequence"].tolist()

        # Handle multilabel targets (arrays) vs scalar targets
        raw_targets = df[target_col].values
        if hasattr(raw_targets[0], "__len__"):
            targets = np.stack(raw_targets).astype(np.float32)
        else:
            targets = raw_targets.astype(np.float32)

        cds = df["cds"].tolist() if "cds" in df.columns else None
        splice = df["splice"].tolist() if "splice" in df.columns else None

        ds = SequenceDataset(sequences, targets, cds, splice)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    train_loader = df_to_loader(splits["train_df"], shuffle=True)
    val_loader = df_to_loader(splits["val_df"], shuffle=False)
    test_loader = df_to_loader(splits["test_df"], shuffle=False)

    return train_loader, val_loader, test_loader


def run_finetune(
    model_class,
    model_version: str,
    dataset: BenchmarkDataset,
    task: str,
    target_col: str,
    split_type: str,
    learning_rate: float,
    lora_rank: int,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    random_seed: int,
    device: torch.device,
    eval_test: bool = False,
) -> tuple[dict[str, float], dict[str, float] | None, dict[str, list[float]]]:
    """Run fine-tuning for a single seed.

    Args:
        model_class: Model class from MODEL_CATALOG.
        model_version: Version of the model.
        dataset: Benchmark dataset.
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
    # Create fine-tunable model
    FineTunableModel = make_fine_tunable(model_class)
    model = FineTunableModel(model_version, device)

    # Apply LoRA
    model.apply_lora(rank=lora_rank)

    # Get embedding dimension and attach head
    dummy_seq = "ATGATG"
    dummy_cds = [np.zeros(len(dummy_seq), dtype=np.float32)]
    dummy_splice = [np.zeros(len(dummy_seq), dtype=np.float32)]
    emb_dim = model.embed([dummy_seq], cds=dummy_cds, splice=dummy_splice).shape[-1]

    # Determine output dimension based on task type
    if task == "regression":
        output_dim = 1
    elif task == "multilabel":
        # Infer from target column shape
        sample_target = dataset.data_df[target_col].iloc[0]
        output_dim = len(sample_target) if hasattr(sample_target, "__len__") else 1
    else:
        # Classification: infer from unique values
        output_dim = dataset.data_df[target_col].nunique()

    head = TaskHead(input_dim=emb_dim, output_dim=output_dim, task_type=task)
    model.attach_head(head)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        target_col=target_col,
        split_type=split_type,
        random_seed=random_seed,
        batch_size=batch_size,
    )

    # Create trainer and fit
    config = TrainerConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        gradient_accumulation_steps=accumulation_steps,
    )
    trainer = FineTuneTrainer(model, config)
    trainer.fit(train_loader, val_loader)

    # Always evaluate on val
    val_metrics = trainer.evaluate(val_loader)

    # Optionally evaluate on test
    test_metrics = None
    if eval_test:
        test_metrics = trainer.evaluate(test_loader)

    return val_metrics, test_metrics, trainer.history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Get model class and version (use default if not specified)
    model_class = MODEL_CATALOG[args.model_name]
    model_version = args.model_version or model_class.default_version
    model_short_name = model_class.get_model_short_name(model_version)

    # Load dataset and infer defaults from metadata
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
    print("Learning rates: {}".format(args.learning_rates))
    print("LoRA ranks: {}".format(args.lora_ranks))
    print("Seeds: {}".format(args.seeds))
    print("Eval test: {}".format(args.eval_test))

    seeds = eval(args.seeds)
    learning_rates = eval(args.learning_rates)
    lora_ranks = eval(args.lora_ranks)

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
