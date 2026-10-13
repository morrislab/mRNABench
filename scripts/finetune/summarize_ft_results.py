"""Summarize fine-tuning results into a CSV."""

import argparse

import pandas as pd

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG
from mrna_bench.persister import ResultsPersister

parser = argparse.ArgumentParser(
    description="Aggregate fine-tuning results into a summary CSV."
)
parser.add_argument(
    "--output", type=str, default="ft_summary.csv",
    help="Output CSV path.",
)
args = parser.parse_args()


if __name__ == "__main__":
    rows = []

    for dataset_name, dataset_cls in DATASET_CATALOG.items():
        try:
            dataset = mb.load_dataset(dataset_name)
        except Exception as e:
            print("Skipping {}: {}".format(dataset_name, e))
            continue

        for result in ResultsPersister.load_all_results(dataset.dataset_path):
            if result["evaluation"] != "fine_tune":
                continue
            row = {
                key: result[key]
                for key in ("model", "task", "target_col", "split_type", "seed")
            }
            row["dataset"] = dataset_name
            row.update(result["embedding_config"])
            row.update(result["probe_config"])
            for split, split_metrics in result["metrics"].items():
                for metric, value in split_metrics.items():
                    row["{}_{}".format(split, metric)] = value
            rows.append(row)

    if not rows:
        print("No fine-tuning results found.")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print("Wrote {} results to {}".format(len(df), args.output))
        print("\nSample:")
        print(df.head().to_string())
