import json
import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt

from mrna_bench.datasets import DATASET_CATALOG


if __name__ == "__main__":
    all_results_mean = {}
    all_results_std = {}

    for dataset_name, ds_class in DATASET_CATALOG.items():
        dataset = ds_class()
        dataset_path = Path(dataset.dataset_path)

        results_mean = {}
        results_std = {}

        results_path = dataset_path / "lp_results"

        if results_path.exists():
            for result_fn in sorted(os.listdir(results_path)):
                model = result_fn.split("_")[3]

                if "250" in model:
                    continue

                overlap = int(result_fn.split("_")[4][1:])

                with open(results_path / result_fn, "r") as f:
                    data = json.load(f)

                model_name = "{}_{}".format(model, overlap)
                if "test_auprc" in data.keys():
                    metric = [float(m) for m in data["test_auprc"].split("±")]
                elif "test_r" in data.keys():
                    metric = [float(m) for m in data["test_r"].split("±")]
                else:
                    raise ValueError

                results_mean[model_name] = metric[0]
                results_std[model_name] = metric[1]

        all_results_mean[dataset_name] = results_mean
        all_results_std[dataset_name] = results_std

    CB_color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00"
    ]

    for dataset_name, metrics in all_results_mean.items():
        metrics = dict(sorted(metrics.items()))

        model_names = ["_".join(k.split("_")[:-1]) for k in metrics.keys()]

        colours = []
        seen_model_class = set()
        c_ind = -1

        # Assumes model_names sorted
        for model_name in model_names:
            model_class = model_name.split("-")[0]
            if model_class not in seen_model_class:
                seen_model_class.add(model_class)
                c_ind += 1
            colours.append(CB_color_cycle[c_ind])

        ax = plt.gca()
        f = plt.gcf()

        x_pos = np.arange(len(model_names))
        plt.bar(x_pos, metrics.values(), color=colours)

        text_anchor = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025

        for i, x in enumerate(x_pos):
            plt.text(
                x - 0.3,
                text_anchor,
                model_names[i],
                rotation=90,
                c="black",
            )

        if dataset_name in ["prot-loc", "go-mf"]:
            plt.ylabel("AUPRC")
        else:
            plt.ylabel("Pearson's R")

        plt.title("Linear Probe " + dataset_name)
        ax.axes.get_xaxis().set_visible(False)
        f.tight_layout()

        plt.savefig("./output/lp_results_{}.png".format(dataset_name))
        plt.close()
