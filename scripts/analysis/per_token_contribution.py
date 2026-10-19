"""Decompose linear probe predictions into exact per-token contributions.

Mean pooling and a linear probe are both linear, so a fitted probe's
prediction decomposes exactly over tokens:

    w . mean(h) + b = (1 / L) * sum_t (w . h_t) + b

This is an identity, not a saliency approximation. It measures how much
of a prediction is produced by a few positions, which is what mean
pooling attenuates by roughly m / L when signal occupies m of L tokens.

Contributions are reported relative to the corpus mean token
contribution. The absolute level of w . h_t is shared across tokens and
absorbed by the intercept, so only the deviation distinguishes one
sequence from another, and only its concentration indicates local
signal.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import mrna_bench as mb
from mrna_bench.fine_tune.poolers import no_pool
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder
from mrna_bench.models import MODEL_CATALOG

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--model_version", type=str, default=None)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
parser.add_argument("--split_type", type=str, default=None)
parser.add_argument("--num_sequences", type=int, default=200)
parser.add_argument("--random_seed", type=int, default=2541)
parser.add_argument("--class_index", type=int, default=0)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--save_profiles", action="store_true")


def get_probe_weights(model, task: str, class_index: int) -> np.ndarray:
    """Extract the probe weight vector used for the decomposition.

    Args:
        model: Fitted sklearn estimator from LinearProbe.
        task: Probing task the estimator was fitted for.
        class_index: Row of coef_ to use for multiclass probes.

    Returns:
        Weight vector of shape (hidden_dim,).

    Raises:
        NotImplementedError: If the estimator is not a single linear
            model, as for multilabel MultiOutputClassifier.
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise NotImplementedError(
            "{} has no coef_; the decomposition needs a single linear "
            "model. Multilabel probes are not supported.".format(
                type(model).__name__
            )
        )

    coef = np.asarray(coef)
    if coef.ndim == 1:
        return coef

    if coef.shape[0] == 1:
        return coef[0]

    print("Multiclass probe: using class_index {} of {}.".format(
        class_index, coef.shape[0]
    ))
    return coef[class_index]


def embed_tokens(model, row) -> np.ndarray:
    """Embed one sequence without pooling.

    Args:
        model: EmbeddingModel instance.
        row: Dataframe row holding the sequence and optional tracks.

    Returns:
        Token embeddings of shape (num_tokens, hidden_dim).
    """
    kwargs = {}
    if "cds" in row and row["cds"] is not None:
        kwargs["cds"] = [np.asarray(row["cds"]).astype(np.int32)]
    if "splice" in row and row["splice"] is not None:
        kwargs["splice"] = [np.asarray(row["splice"]).astype(np.int32)]

    tokens = model.embed([row["sequence"]], agg_fn=no_pool, **kwargs)

    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(
            "Expected (1, num_tokens, hidden) from no_pool, got {}. This "
            "model's wrapper aggregates over chunks before agg_fn and "
            "cannot yield token embeddings.".format(tuple(tokens.shape))
        )

    return tokens[0].float().detach().cpu().numpy()


def summarize_concentration(deviations: np.ndarray) -> dict:
    """Measure how concentrated per-token contributions are.

    Args:
        deviations: Per-token contributions relative to the corpus mean.

    Returns:
        Dictionary of concentration statistics for one sequence.
    """
    magnitude = np.abs(deviations)
    total = magnitude.sum()
    num_tokens = len(deviations)

    if total == 0:
        return {
            "num_tokens": num_tokens,
            "effective_positions": float(num_tokens),
            "effective_fraction": 1.0,
            "top1pct_mass": 0.0,
            "top5pct_mass": 0.0,
        }

    # Participation ratio: effective count of contributing positions,
    # equal to num_tokens when every token contributes equally.
    effective = total ** 2 / np.square(deviations).sum()

    ordered = np.sort(magnitude)[::-1]
    top1 = max(1, int(round(0.01 * num_tokens)))
    top5 = max(1, int(round(0.05 * num_tokens)))

    return {
        "num_tokens": num_tokens,
        "effective_positions": float(effective),
        "effective_fraction": float(effective / num_tokens),
        "top1pct_mass": float(ordered[:top1].sum() / total),
        "top5pct_mass": float(ordered[:top5].sum() / total),
    }


if __name__ == "__main__":
    args = parser.parse_args()

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

    print("Dataset: {}".format(args.dataset_name))
    print("Model: {}".format(model_short_name))
    print("Task: {} on {}".format(task, target))

    # Fit the probe on the cached mean-pooled embeddings, matching the
    # reported benchmark path, then reuse its weights.
    probe = (
        LinearProbeBuilder(dataset)
        .fetch_embedding_by_model_name(model_short_name)
        .build_splitter(split_type)
        .set_target(target)
        .build_evaluator(task)
        .build()
    )
    metrics = probe.run_linear_probe(random_seed=args.random_seed)
    print("Probe metrics: {}".format(metrics))

    fitted = probe.models[args.random_seed]
    weights = get_probe_weights(fitted, task, args.class_index)

    embedder = mb.load_model(args.model_name, model_version, device)
    embedder.set_inference_mode()

    sample_df = dataset.data_df.dropna(subset=[target])
    if args.num_sequences < len(sample_df):
        sample_df = sample_df.sample(
            n=args.num_sequences, random_state=args.random_seed
        )

    print("Embedding {} sequences without pooling.".format(len(sample_df)))

    contributions = []
    identity_errors = []
    cache_errors = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        tokens = embed_tokens(embedder, row)
        per_token = tokens @ weights
        contributions.append(per_token)

        pooled_here = float(tokens.mean(axis=0) @ weights)
        identity_errors.append(abs(pooled_here - float(per_token.mean())))

        # LinearProbe attaches the cached mean-pooled embeddings to
        # data_df, so mean over no_pool tokens must reproduce them.
        if "embeddings" in row:
            cached = np.asarray(row["embeddings"], dtype=np.float64)
            cache_errors.append(
                float(np.abs(tokens.mean(axis=0) - cached).max())
            )

    corpus_mean = np.mean(np.concatenate(contributions))

    records = []
    for (_, row), per_token in zip(sample_df.iterrows(), contributions):
        deviations = per_token - corpus_mean
        stats = summarize_concentration(deviations)
        stats["sequence_length"] = len(row["sequence"])
        stats["prediction_deviation"] = float(deviations.mean())
        records.append(stats)

    fractions = np.array([r["effective_fraction"] for r in records])
    lengths = np.array([r["sequence_length"] for r in records])
    top1 = np.array([r["top1pct_mass"] for r in records])

    # Concentration detects a determinant being present, not absent:
    # a sequence lacking the motif has nothing to localize and reads as
    # diffuse however strongly it deviates. So split by signed
    # deviation rather than magnitude, and report both tails, since
    # which one carries the local feature is not known in advance.
    signed = np.array([r["prediction_deviation"] for r in records])
    upper_tail = signed >= np.quantile(signed, 0.75)
    lower_tail = signed <= np.quantile(signed, 0.25)

    print("\nDecomposition identity max error: {:.2e}".format(
        max(identity_errors)
    ))
    if cache_errors:
        print("Token-mean vs cached embedding max error: {:.2e}".format(
            max(cache_errors)
        ))
        print("  A large value means no_pool tokens do not reproduce the "
              "cached embeddings, so the decomposition is not measuring "
              "the probed representation.")
    print("\nLocalization across {} sequences:".format(len(records)))
    print("  effective fraction, all:          {:.4f}".format(
        float(np.median(fractions))
    ))
    print("  effective fraction, upper tail:   {:.4f}".format(
        float(np.median(fractions[upper_tail]))
    ))
    print("  effective fraction, lower tail:   {:.4f}".format(
        float(np.median(fractions[lower_tail]))
    ))
    print("  top 1% mass share, all:           {:.4f}".format(
        float(np.median(top1))
    ))
    print("  top 1% mass share, upper tail:    {:.4f}".format(
        float(np.median(top1[upper_tail]))
    ))
    print("  top 1% mass share, lower tail:    {:.4f}".format(
        float(np.median(top1[lower_tail]))
    ))
    print("  A tail well below the others localizes signal in few")
    print("  positions, which is what mean pooling dilutes by m / L.")
    print("  median tokens per sequence: {:.0f}".format(
        float(np.median([r["num_tokens"] for r in records]))
    ))

    if upper_tail.sum() > 2 and np.std(lengths[upper_tail]) > 0:
        corr = float(np.corrcoef(
            np.log(lengths[upper_tail]), fractions[upper_tail]
        )[0, 1])
        print("  corr(log length, effective fraction): {:.3f}".format(corr))
        print("  Negative correlation means longer sequences concentrate")
        print("  signal in fewer positions, so mean pooling dilutes more.")

    out_dir = Path(args.out_dir or Path(dataset.dataset_path) / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = "pertoken_{}_{}_{}_rs-{}".format(
        args.dataset_name, model_short_name, target, args.random_seed
    )

    summary = {
        "dataset": args.dataset_name,
        "model": model_short_name,
        "task": task,
        "target_col": target,
        "split_type": split_type,
        "random_seed": args.random_seed,
        "probe_metrics": metrics,
        "decomposition_max_error": float(max(identity_errors)),
        "cache_max_error": (
            float(max(cache_errors)) if cache_errors else None
        ),
        "median_effective_fraction": float(np.median(fractions)),
        "median_top1pct_mass": float(np.median(top1)),
        "upper_tail_effective_fraction": float(
            np.median(fractions[upper_tail])
        ),
        "lower_tail_effective_fraction": float(
            np.median(fractions[lower_tail])
        ),
        "upper_tail_top1pct_mass": float(np.median(top1[upper_tail])),
        "lower_tail_top1pct_mass": float(np.median(top1[lower_tail])),
        "per_sequence": records,
    }

    with open(out_dir / (stem + ".json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if args.save_profiles:
        np.savez_compressed(
            out_dir / (stem + "_profiles.npz"),
            corpus_mean=corpus_mean,
            **{
                "seq_{}".format(i): c
                for i, c in enumerate(contributions)
            },
        )

    print("\nWrote {}".format(out_dir / (stem + ".json")))
