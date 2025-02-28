import argparse
import torch
import numpy as np
import mrna_bench as mb
import pandas as pd
from collections import defaultdict
from mrna_bench.embedder import DatasetEmbedder, get_output_filepath
from mrna_bench.datasets import DATASET_INFO
from mrna_bench.linear_probe import LinearProbe
import os

def main():
    # --- Parse arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=False,
                        help="Directory containing model checkpoints")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory to write per-model results")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite of existing results")
    parser.add_argument("--default-naming", action="store_true",
                        help="Assume custom model directory naming")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name (if not using custom naming)")
    parser.add_argument("--model-version", type=str, default=None,
                        help="Model version (if not using custom naming)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file (if not using custom naming)")
    parser.add_argument("--recompute-embeddings", action="store_true",
                        help="Recompute embeddings for all datasets")
    parser.add_argument("--sequence-chunk-overlap", type=int, default=0,
                        help="Amount of overlap between sequence chunks")
    args = parser.parse_args()

    if args.default_naming:

        if args.model_dir is None:
            raise ValueError("Model directory must be provided if using default naming (assumes Orthrus model directory format)")
        # the naming for the directory is in the format:
        # {model_name}_{model_version}

        # we extract the model name (and capitalize it) and the version is just the directory name
        model_name = os.path.basename(args.model_dir).split("_")[0].capitalize()
        model_version = os.path.basename(args.model_dir)

        # the checkpoint that we use is the one with the latest epoch value
        # checkpoint naming format: epoch={epoch}-step={step}.ckpt
        checkpoint_files = os.listdir(args.model_dir)
        checkpoint_files = [f for f in checkpoint_files if f.endswith(".ckpt")]
        latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split("-")[0].split("=")[1]), reverse=True)[0]
    else:
        # if we're not using custom naming, we need to provide the model name and version,
        if args.model_name is None or args.model_version is None:
            raise ValueError("Model name and version must be provided if not using custom naming")

        model_name = args.model_name.capitalize()
        model_version = args.model_version
        latest_checkpoint = args.checkpoint if args.checkpoint is not None else None
    

    print(f"Probing model {model_version}...")

    # Create a new column order list that will have four columns per dataset.
    agg_col_order = []
    seed_col_order = []
    for ds, info in DATASET_INFO.items():
        col_name = info["col_name"]
        if info["task"] in ["classification", "multilabel"]:
            agg_col_order.extend([f"{col_name}_AUROC", f"{col_name}_AUROC_std",
                                  f"{col_name}_AUPRC", f"{col_name}_AUPRC_std"])
            seed_col_order.extend([f"{col_name}_AUROC", f"{col_name}_AUPRC"])
        else:
            agg_col_order.extend([f"{col_name}_MSE", f"{col_name}_MSE_std",
                                  f"{col_name}_R", f"{col_name}_R_std"])
            seed_col_order.extend([f"{col_name}_MSE", f"{col_name}_R"])

    agg_out_path = os.path.join(args.results_dir, f"{model_version}_aggregated_results.tsv")
    per_seed_out_path = os.path.join(args.results_dir, f"{model_version}_per_seed_results.tsv")

    print(f"Determining columns for {model_version}...")
    if args.force:
        print("Force overwrite enabled. Deleting existing results...")
        if os.path.exists(agg_out_path):
            os.remove(agg_out_path)
        if os.path.exists(per_seed_out_path):
            os.remove(per_seed_out_path)

    # before we start, we should check if the aggregated output files already exist, and if so, check which datasets (if any) are missing
    if os.path.exists(agg_out_path) and os.path.exists(per_seed_out_path):
        results_df = pd.read_csv(agg_out_path, sep="\t")

        datasets_to_remove = []
        for dataset_name in DATASET_INFO.keys():
            col_name = DATASET_INFO[dataset_name]["col_name"] + ("_AUROC" if DATASET_INFO[dataset_name]["task"] in ["classification", "multilabel"] else "_R")
            if col_name in results_df.columns:
                # print(f"Results for dataset {dataset_name} already exist in {agg_out_path}. Skipping...")
                datasets_to_remove.append(dataset_name)
        
        for dataset_name in datasets_to_remove:
            del DATASET_INFO[dataset_name]

    # don't run if all datasets have already been probed and we're not forcing overwrite
    if not args.force and len(DATASET_INFO) == 0:
        print(f"All datasets already probed in {agg_out_path}. Exiting...")
        return

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dictionaries for aggregated (average) results and per-seed results.
    aggregated_results = defaultdict(dict)
    # per_seed_results: key is seed, value is a dict with "Model" and one column per dataset.
    seeds = [0, 1, 7, 9, 42, 123, 256, 777, 2025, 31415]
    per_seed_results = {s: {"Model": model_version, "seed": s} for s in seeds}

    # Run probing for each dataset sequentially
    for dataset_name in DATASET_INFO.keys():
        print(f"Evaluating model {model_version} on dataset {dataset_name}...")
        
        dataset = mb.load_dataset(
            DATASET_INFO[dataset_name]["dataset"], 
            isoform_resolved=DATASET_INFO[dataset_name]["isoform_resolved"],
            target_col=DATASET_INFO[dataset_name]["target_col"]
        )

        model = mb.load_model(
            model_name = model_name, 
            model_version = model_version, 
            checkpoint = latest_checkpoint,
            device = device
        )

        # checking if the embeddings already exist
        model_short_name = model.get_model_short_name(model_version)

        embeddings_path = get_output_filepath(
            output_dir = dataset.embedding_dir,
            model_short_name = model_short_name,
            dataset_name = DATASET_INFO[dataset_name]["dataset"],
            sequence_chunk_overlap = args.sequence_chunk_overlap
        ) + ".npz"

        if args.recompute_embeddings and os.path.exists(embeddings_path):
            print(f"Recomputing embeddings for {model_version} on {dataset_name}...")
            os.remove(embeddings_path)

        # we only want to embed the sequences once if it's the same dataset (just different targets)
        if os.path.exists(embeddings_path):
            print(f"Embeddings already exist for {model_version} on {dataset_name}. Loading...")
            embeddings = np.load(embeddings_path)["embedding"]
        else:
            embedder = DatasetEmbedder(model, dataset, transcript_avg=DATASET_INFO[dataset_name]["transcript_avg"])
            embeddings = embedder.embed_dataset()
            embedder.persist_embeddings(embeddings)

            # detach and move to cpu to save memory
            embeddings = embeddings.detach().cpu().numpy()

        print(f"Prober initialized for model for {model_version} on {dataset_name}")
        prober = LinearProbe(
            model_short_name=model_name,
            seq_chunk_overlap=args.sequence_chunk_overlap,
            dataset=dataset,
            embeddings=embeddings,
            task=DATASET_INFO[dataset_name]["task"],
            target_col=DATASET_INFO[dataset_name]["target_col"],
            split_type=DATASET_INFO[dataset_name]["split_type"],
            # eval_all_splits=True,
            ss_map_path="/home/dalalt1/Orthrus_eval/essentiality/lncRNA_homology/output/similarity_results_full.npz",
            threshold=0.75,
        )

        # Run multi-run probing: metrics is a dict with key=seed and value=dict of metric values.
        metrics = prober.linear_probe_multirun(random_seeds=seeds)
        average_metrics = prober.compute_multirun_results(metrics)
        aggregated_results[model_version][dataset_name] = average_metrics

        print(f"Results for {model_version} on {dataset_name}: {average_metrics}")

        col_name = DATASET_INFO[dataset_name]["col_name"]
        # Save per-seed results with separate columns.
        for s in seeds:
            if DATASET_INFO[dataset_name]["task"] in ["classification", "multilabel"]:
                # For classification tasks
                auroc = float(metrics[s].get("val_auroc", np.nan))
                auprc = float(metrics[s].get("val_auprc", np.nan))
                try:
                    # For a single run, std is not defined.
                    per_seed_results[s][f"{col_name}_AUROC"] = round(auroc, 4)
                    per_seed_results[s][f"{col_name}_AUPRC"] = round(auprc, 4)
                except Exception:
                    per_seed_results[s][f"{col_name}_AUROC"] = auroc
                    per_seed_results[s][f"{col_name}_AUPRC"] = auprc
            else:
                # For regression tasks
                mse = float(metrics[s].get("val_mse", np.nan))
                r = float(metrics[s].get("val_r", np.nan))
                try:
                    per_seed_results[s][f"{col_name}_MSE"] = round(mse, 4)
                    per_seed_results[s][f"{col_name}_R"] = round(r, 4)
                except Exception:
                    per_seed_results[s][f"{col_name}_MSE"] = mse
                    per_seed_results[s][f"{col_name}_R"] = r

        # Save aggregated (averaged) results with separate columns.
        avg_metrics = aggregated_results[model_version][dataset_name]
        # If the task is classification, we expect keys "val_auroc" and "val_auprc"
        if DATASET_INFO[dataset_name]["task"] in ["classification", "multilabel"]:
            auroc_str = avg_metrics.get("val_auroc", "")
            auprc_str = avg_metrics.get("val_auprc", "")
            try:
                auroc = float(auroc_str.split(" ± ")[0])
                auroc_err = float(auroc_str.split(" ± ")[1])
                auprc = float(auprc_str.split(" ± ")[0])
                auprc_err = float(auprc_str.split(" ± ")[1])
                # Save separately in the aggregated results dictionary.
                aggregated_results[model_version][dataset_name] = {
                    f"{col_name}_AUROC": round(auroc, 4),
                    f"{col_name}_AUROC_std": round(auroc_err, 4),
                    f"{col_name}_AUPRC": round(auprc, 4),
                    f"{col_name}_AUPRC_std": round(auprc_err, 4)
                }
            except Exception:
                aggregated_results[model_version][dataset_name] = {
                    f"{col_name}_AUROC": auroc_str,
                    f"{col_name}_AUROC_std": "",
                    f"{col_name}_AUPRC": auprc_str,
                    f"{col_name}_AUPRC_std": ""
                }
        else:
            mse_str = avg_metrics.get("val_mse", np.nan)
            r_str = avg_metrics.get("val_r", np.nan)
            try:
                mse = float(mse_str.split(" ± ")[0])
                mse_err = float(mse_str.split(" ± ")[1])
                r = float(r_str.split(" ± ")[0])
                r_err = float(r_str.split(" ± ")[1])
                aggregated_results[model_version][dataset_name] = {
                    f"{col_name}_MSE": round(mse, 4),
                    f"{col_name}_MSE_std": round(mse_err, 4),
                    f"{col_name}_R": round(r, 4),
                    f"{col_name}_R_std": round(r_err, 4)
                }
            except Exception:
                aggregated_results[model_version][dataset_name] = {
                    f"{col_name}_MSE": mse_str,
                    f"{col_name}_MSE_std": "",
                    f"{col_name}_R": r_str,
                    f"{col_name}_R_std": ""
                }

    # Build the aggregated results row.
    agg_row = {"Model": model_version}
    # For every dataset, update agg_row with the four entries.
    for ds, info in DATASET_INFO.items():
        col_name = info["col_name"]
        if ds in aggregated_results[model_version]:
            ds_results = aggregated_results[model_version][ds]
            for key, val in ds_results.items():
                agg_row[key] = val
        else:
            # Fill with "N/A" if not available.
            if info["task"] in ["classification", "multilabel"]:
                agg_row[f"{col_name}_AUROC"] = np.nan
                agg_row[f"{col_name}_AUROC_std"] = np.nan
                agg_row[f"{col_name}_AUPRC"] = np.nan
                agg_row[f"{col_name}_AUPRC_std"] = np.nan
            else:
                agg_row[f"{col_name}_MSE"] = np.nan
                agg_row[f"{col_name}_MSE_std"] = np.nan
                agg_row[f"{col_name}_R"] = np.nan
                agg_row[f"{col_name}_R_std"] = np.nan

    agg_cols = ["Model"] + agg_col_order
    results_df = pd.DataFrame([agg_row], columns=agg_cols)

    # per-seed results DataFrame.
    seed_cols = ["Model", "seed"] + seed_col_order
    per_seed_df = pd.DataFrame([per_seed_results[s] for s in seeds], columns=seed_cols)

    os.makedirs(args.results_dir, exist_ok=True)

    if os.path.exists(agg_out_path) and os.path.exists(per_seed_out_path):
        original_results_df = pd.read_csv(agg_out_path, sep="\t")
        results_df = pd.merge(original_results_df, results_df, on="Model")
        
        original_per_seed_df = pd.read_csv(per_seed_out_path, sep="\t")
        per_seed_df = pd.merge(original_per_seed_df, per_seed_df, on=["Model", "seed"])

    # correct col order
    results_df = results_df[['Model'] + agg_col_order]
    per_seed_df = per_seed_df[['Model', 'seed'] + seed_col_order]

    results_df.to_csv(agg_out_path, sep="\t", index=False)
    per_seed_df.to_csv(per_seed_out_path, sep="\t", index=False)

    print(f"Aggregated results for {model_version} saved to {agg_out_path}")
    print(f"Per-seed results for {model_version} saved to {per_seed_out_path}")

if __name__ == "__main__":
    main()