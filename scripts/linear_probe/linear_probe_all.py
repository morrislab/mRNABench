import argparse
import torch
import mrna_bench as mb
import pandas as pd
from collections import defaultdict
from mrna_bench.embedder import DatasetEmbedder
from mrna_bench.linear_probe import LinearProbe
from mrna_bench.datasets import DATASET_INFO  # DATASET_INFO contains dataset parameters

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description="Run linear probing across all datasets in DATASET_INFO using embeddings from a given model."
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model class (e.g., Orthrus)",
)
parser.add_argument(
    "--model_version",
    type=str,
    required=True,
    help="Version or full config string for the model",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    required=True,
    help="Checkpoint filename (e.g., epoch=22-step=20000.ckpt)",
)
parser.add_argument(
    "--mrna_bench_data_storage",
    type=str,
    default="/home/fradkinp/Documents/01_projects/data_storage",
    help="Directory to save results",
)
parser.add_argument(
    "--mask_out_splice_track",
    type=str2bool,
    default=False,
    help="Mask out splice track",
)
parser.add_argument(
    "--mask_out_cds_track",
    type=str2bool,
    default=False,
    help="Mask out CDS track",
)
args = parser.parse_args()

# pretty print args
print(args)


# Set computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define seeds for reproducibility
seeds = [0, 1, 7, 9, 42, 123, 256, 777, 2025, 31415]

# Dictionary to collect results (keyed by model version then dataset name)
results = defaultdict(dict)

# Iterate over all datasets from DATASET_INFO
for dataset_name, params in DATASET_INFO.items():
    print(f"Evaluating model {args.model_version} on dataset '{dataset_name}'...")

    # Load the dataset.
    # DATASET_INFO is assumed to have keys like 'dataset', 'target_col', 'task',
    # 'split_type', 'isoform_resolved', and 'transcript_avg'
    if args.mask_out_splice_track or args.mask_out_cds_track:
        force_redownload = True
    else:
        force_redownload = False


    dataset = mb.load_dataset(
        params["dataset"],
        isoform_resolved=params.get("isoform_resolved", False),
        target_col=params["target_col"],
        mask_out_splice_track=args.mask_out_splice_track,
        mask_out_cds_track=args.mask_out_cds_track,
        force_redownload=force_redownload
    )

    # Load the model using the provided checkpoint and device.
    model = mb.load_model(
        model_name=args.model_name,
        model_version=args.model_version,
        checkpoint=args.ckpt_name,
        device=device,
    )

    # Embed the dataset (pass transcript_avg if specified)
    embedder = DatasetEmbedder(model, dataset, transcript_avg=params.get("transcript_avg", False))
    embeddings = embedder.embed_dataset().detach().cpu().numpy()

    # If the dataset name indicates essentiality, filter the dataset accordingly
    if "ess" in dataset_name:
        dataset.data_df = dataset.data_df[dataset.data_df.isoform_resolved == 1].copy()

    # Create a LinearProbe instance with dataset-specific parameters.
    prober = LinearProbe(
        model_short_name=args.model_name,
        seq_chunk_overlap=0,
        dataset=dataset,
        embeddings=embeddings,
        task=params["task"],
        target_col=params["target_col"],
        split_type=params.get("split_type", "default"),
        ss_map_path="/data1/morrisq/dalalt1/Orthrus/processed_data/essentiality/sanjana_data/miscellaneous/similarity_results_full.npz",
        threshold=0.9,
    )

    # Run multi-run probing with the specified seeds.
    metrics = prober.linear_probe_multirun(random_seeds=seeds)
    average_metrics = prober.compute_multirun_results(metrics)
    results[args.model_version][dataset_name] = average_metrics

    print(average_metrics)

all_rows = {}
for model_version, dataset_dict in results.items():
    # If there's only one dataset per model_version, grab its metrics
    row = {}
    for dataset_name, combined_val in dataset_dict.items():
        for metric_name, metric_vals in combined_val.items():
            if "±" in metric_vals:
                mean_str, se_str = metric_vals.split("±")
                mean_val = float(mean_str.strip())
                se_val = float(se_str.strip())
                row[f"{dataset_name}_{metric_name}_mean"] = mean_val
                row[f"{dataset_name}_{metric_name}_se"] = se_val
            else:
                row[f"{dataset_name}_{metric_name}"] = metric_vals
    all_rows[model_version] = row
# Create a DataFrame with model_version as the index.
df = pd.DataFrame.from_dict(all_rows, orient='index')

# Convert to a pandas DataFrame and save as CSV.
run_name = f"{args.model_version}_{args.ckpt_name}_mask_splice_{args.mask_out_splice_track}_mask_cds_{args.mask_out_cds_track}"
output_path = f"{args.mrna_bench_data_storage}/mrna_bench_results2/{run_name}.csv"
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
