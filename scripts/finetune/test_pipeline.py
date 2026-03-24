"""Quick test of fine-tuning pipeline."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from mrna_bench import load_dataset
from mrna_bench.fine_tune import (
    FineTunePersister,
    FineTuneTrainer,
    FineTuneWrapper,
    SequenceDataset,
    TaskHead,
    TrainerConfig,
    collate_fn,
)
from mrna_bench.models import MODEL_CATALOG


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Load real dataset for persister
    dataset = load_dataset("rnahl-human")
    task = dataset.metadata.task[0]
    target_col = dataset.metadata.target_col[0]
    split_type = dataset.metadata.default_split_type

    print("Dataset: {}".format(dataset.dataset_name))
    print("Task: {}".format(task))
    print("Target: {}".format(target_col))

    # Use RNA-FM
    model_class = MODEL_CATALOG["RNA-FM"]
    model_version = model_class.default_version
    model_short_name = model_class.get_model_short_name(model_version)

    model = model_class(model_version, device)
    model.set_inference_mode()

    lora_rank = 4
    learning_rate = 1e-4

    # Get embedding dim
    emb_dim = model.embed(["AUGC"]).shape[-1]
    print("Embedding dim: {}".format(emb_dim))

    # Create wrapper with head and LoRA
    head = TaskHead(input_dim=emb_dim, output_dim=1, task_type=task)
    wrapper = FineTuneWrapper(model, head)
    wrapper.apply_lora(rank=lora_rank)

    param_info = wrapper.get_parameter_count()
    print("Parameters: {}".format(param_info))

    # Create tiny synthetic dataset for quick test
    sequences = ["AUGCAUGCAUGC" * 10] * 20
    targets = np.random.randn(20).astype(np.float32)

    train_ds = SequenceDataset(sequences[:16], targets[:16])
    val_ds = SequenceDataset(sequences[16:], targets[16:])

    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)

    # Train for 2 epochs
    config = TrainerConfig(
        learning_rate=learning_rate,
        epochs=2,
        early_stopping_patience=5,
    )
    trainer = FineTuneTrainer(wrapper, config)
    history = trainer.fit(train_loader, val_loader)

    print("\nHistory: {}".format(history))

    # Evaluate
    val_metrics = trainer.evaluate(val_loader)
    print("\nVal metrics: {}".format(val_metrics))

    # Test persister
    persister = FineTunePersister(
        dataset=dataset,
        model_short_name=model_short_name,
        task=task,
        target_col=target_col,
        split_type=split_type,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
    )

    random_seed = "test"
    persister.persist_run_results(
        metrics={"val": val_metrics, "test": None},
        random_seed=random_seed,
        history=history,
    )
    result_path = persister._get_path(random_seed, ".json")
    print("\nPersisted results to: {}".format(result_path))

    # Load back and verify
    loaded = persister.load_run_results(random_seed)
    print("Loaded results: {}".format(loaded))
