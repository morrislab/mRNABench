# Result Aggregation

This directory contains scripts for aggregating and analyzing results from the linear probing pipeline.

## `aggregate_orthrus_lp.py`

This is the primary script for post-processing linear probe results. It scans all registered datasets, finds the JSON output files from the probing runs, and provides several modes for summarizing and analyzing the data.

### Core Functionality

The script's main purpose is to aggregate raw, per-seed metrics into a summary table.

- **Aggregation**: By default, it calculates the mean and standard deviation of all metrics across all random seeds for each unique combination of model, dataset, task, target, and split.
- **Filtering**: You can limit which models or seeds are included in the aggregation using the `--config_file` and `--seeds` flags.
- **Output Format**: The default output is a "long-format" CSV printed to the console. You can save the output to a file with `--output_filename` and pivot the data into a "wide-format" table (one row per model) using the `--wide_format` flag.

### Z-Score Analysis

The script can normalize model performance by calculating Z-scores for each metric relative to a reference group of models. A Z-score indicates how many standard deviations a model's performance is from the reference mean.

- **Activation**: This analysis is activated by providing a reference model set and specifying an output file.
- **Reference Group**: The reference statistics (mean and standard deviation) can be calculated from either a group of models specified in a JSON/YAML file (`--zscore_ref_config_file`) or a single model (`--zscore_ref_model_name`).
- **Output**: Use `--z_score_output` to provide a filename for the Z-score report. The report shows the average Z-score for each model across all seeds for the key metrics (`auprc`, `auroc`, `r`).

### Statistical Significance Testing

To determine if a group of models ("test" group) is statistically different from a reference group, the script can perform an independent t-test (Welch's t-test) on their Z-score distributions.

- **Activation**: This analysis is activated by providing a reference for the test and specifying an output file.
- **Test Group**: The models to be evaluated are specified using the main `--config_file` argument.
- **Reference Group**: The reference group for the t-test is specified independently using either `--sig_ref_config_file` or `--sig_ref_model_name`.
- **Output**: Use `--significance_output` to provide a filename for the significance report, which will contain p-values for each metric, indicating the significance of the difference between the test and reference groups.

### Example Usage

The following command demonstrates a complete analysis workflow:
1.  Aggregates results for models listed in `ablation_runs.json`.
2.  Generates a wide-format summary CSV named `architecture_ablation.csv`.
3.  Calculates Z-scores for all models using the same `ablation_runs.json` as the reference and saves them to `architecture_ablation_z_score.csv`.
4.  Performs a significance test comparing all models in `ablation_runs.json` against a single reference model (`ssm_6t_...`) and saves the results to `architecture_ablation_significance.csv`.

```bash
python aggregate_orthrus_lp.py \
--config_file ../model_json_files/ablation_runs.json \
--wide_format \
--output_filename architecture_ablation.csv \
--zscore_ref_config_file ../model_json_files/ablation_runs.json \
--z_score_output architecture_ablation_z_score.csv \
--sig_ref_model_name "ssm-6t-6-512-lr0.001-wd5e-05-mask0.3-new-splice-to-toga-combined-30-mask-1-ablation-epoch=6-step=20000" \
--significance_output architecture_ablation_significance.csv
``` 