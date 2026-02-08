"""Regression test for mRNABench linear probe pipeline.

This script tests the full embedding and linear probing pipeline
to ensure model changes don't cause performance regressions.
"""

import argparse
import sys

import numpy as np
import torch

import mrna_bench as mb
from mrna_bench.embedder import DatasetEmbedder
from mrna_bench.linear_probe import LinearProbeBuilder


DEFAULT_DATASET = "mrl-sugimoto"
DEFAULT_MODEL = "RNA-FM"
DEFAULT_TASK = "regression"
DEFAULT_SPLIT = "default"
DEFAULT_TARGET = "target"
DEFAULT_N_SAMPLES = 0
DEFAULT_SEED = 2541


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Regression test for mRNABench pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset to use for testing (default: {})".format(DEFAULT_DATASET)
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model to use for testing (default: {})".format(DEFAULT_MODEL)
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version (default: model's default version)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Linear probe task type (default: {})".format(DEFAULT_TASK)
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Split type (default: {})".format(DEFAULT_SPLIT)
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Target column (default: {})".format(DEFAULT_TARGET)
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of samples to use (default: {})".format(DEFAULT_N_SAMPLES)
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: {})".format(DEFAULT_SEED)
    )
    parser.add_argument(
        "--expected-metric",
        type=float,
        default=None,
        help="Expected validation metric value for regression check"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for metric comparison (default: 0.01)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    return parser.parse_args()


def run_regression_test(
    dataset_name: str,
    model_name: str,
    model_version: str,
    task: str,
    split_type: str,
    target_col: str,
    n_samples: int,
    seed: int,
    expected_metric: float = None,
    tolerance: float = 0.01,
    verbose: bool = False,
) -> dict:
    """Run the full embedding and linear probe pipeline.

    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model to use.
        model_version: Version of the model.
        task: Linear probe task type.
        split_type: Type of data split.
        target_col: Target column for probing.
        n_samples: Number of samples to use from dataset.
        seed: Random seed for reproducibility.
        expected_metric: Expected metric value for regression check.
        tolerance: Tolerance for metric comparison.
        verbose: Print verbose output.

    Returns:
        Dictionary containing metrics from linear probe.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("Loading dataset: {}".format(dataset_name))

    dataset = mb.load_dataset(dataset_name)

    if n_samples > 0 and n_samples < len(dataset.data_df):
        if verbose:
            print("Subsampling to {} samples".format(n_samples))
        dataset.data_df = dataset.data_df.sample(
            n=n_samples,
            random_state=seed
        ).reset_index(drop=True)

    if verbose:
        print("Dataset size: {}".format(len(dataset.data_df)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device: {}".format(device))

    if verbose:
        print("Loading model: {} ({})".format(model_name, model_version))

    model = mb.load_model(model_name, model_version, device)

    if verbose:
        print("Embedding dataset...")

    embedder = DatasetEmbedder(model, dataset)
    embeddings = embedder.embed_dataset()
    embeddings = embeddings.detach().cpu().numpy()

    if verbose:
        print("Embedding shape: {}".format(embeddings.shape))

    if verbose:
        print("Building linear probe...")

    prober = (
        LinearProbeBuilder(dataset)
        .fetch_embedding_by_embedding_instance(model.short_name, embeddings)
        .build_splitter(split_type, eval_all_splits=False)
        .build_evaluator(task)
        .set_target(target_col)
        .build()
    )

    if verbose:
        print("Running linear probe with seed {}...".format(seed))

    metrics = prober.run_linear_probe(seed)

    if verbose:
        print("Metrics: {}".format(metrics))

    if expected_metric is not None:
        metric_key = "val_r"
        if metric_key not in metrics:
            metric_key = list(metrics.keys())[0]

        actual = metrics[metric_key]
        diff = abs(actual - expected_metric)

        if diff > tolerance:
            msg = "REGRESSION DETECTED: Expected {} = {}, got {} (diff: {})"
            print(msg.format(metric_key, expected_metric, actual, diff))
            return {"status": "fail", "metrics": metrics, "diff": diff}
        else:
            if verbose:
                print(
                    "PASSED: {} = {} (expected {}, diff: {})".format(
                        metric_key, actual, expected_metric, diff
                    )
                )

    return {"status": "pass", "metrics": metrics}


def main():
    """Main entry point."""
    args = parse_args()

    version_str = args.model_version if args.model_version else "default"

    print("=" * 60)
    print("mRNABench Regression Test")
    print("=" * 60)
    print("Dataset: {}".format(args.dataset))
    print("Model: {} ({})".format(args.model, version_str))
    print("Task: {}".format(args.task))
    print("Split: {}".format(args.split))
    print("N samples: {}".format(args.n_samples))
    print("Seed: {}".format(args.seed))
    print("=" * 60)

    result = run_regression_test(
        dataset_name=args.dataset,
        model_name=args.model,
        model_version=args.model_version,
        task=args.task,
        split_type=args.split,
        target_col=args.target,
        n_samples=args.n_samples,
        seed=args.seed,
        expected_metric=args.expected_metric,
        tolerance=args.tolerance,
        verbose=args.verbose,
    )

    print("=" * 60)
    print("Results:")
    for key, value in result["metrics"].items():
        print("  {}: {:.6f}".format(key, value))
    print("=" * 60)

    if result["status"] == "fail":
        print("TEST FAILED - Regression detected!")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
