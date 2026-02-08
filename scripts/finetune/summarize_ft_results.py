"""Summarize fine-tuning results into a CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG

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

        ft_dir = Path(dataset.dataset_path) / "ft_results"
        if not ft_dir.exists():
            continue

        for json_path in sorted(ft_dir.glob("result_ft_*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print("Skipping {}: {}".format(json_path.name, e))
                continue

            config = data.get("config", {})
            metrics = data.get("metrics", {})

            row = dict(config)

            val_metrics = metrics.get("val", {})
            if val_metrics:
                for k, v in val_metrics.items():
                    row["val_{}".format(k)] = v

            test_metrics = metrics.get("test", {})
            if test_metrics:
                for k, v in test_metrics.items():
                    row["test_{}".format(k)] = v

            rows.append(row)

    if not rows:
        print("No fine-tuning results found.")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print("Wrote {} results to {}".format(len(df), args.output))
        print("\nSample:")
        print(df.head().to_string())
