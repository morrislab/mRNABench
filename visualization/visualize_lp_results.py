import json
import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.models.model_catalog import MODEL_CATALOG, MODEL_VERSION_MAP


def get_model_shortname_class_map() -> dict[str, str]:
    """Construct map between model short name and model class."""
    model_shortname_map = {}
    for model_name, model_class in MODEL_CATALOG.items():
        for model_version in MODEL_VERSION_MAP[model_name]:
            short_name = model_class.get_model_short_name(model_version)

            model_shortname_map[short_name] = model_name

    return model_shortname_map


def sort_embedding_by_model_class(embedding_names: list[str]) -> list[str]:
    """Return list of embeddings sorted by model class name."""
    shortname_map = get_model_shortname_class_map()
    output = []

    embedding_map = {}

    for embedding_name in embedding_names:
        model_short_name = embedding_name.split("_")[3]

        model_class = shortname_map[model_short_name]
        embedding_map.setdefault(model_class, []).append(embedding_name)

    embedding_map = dict(sorted(embedding_map.items()))

    for v in embedding_map.values():
        output.extend(sorted(v))

    return output


if __name__ == "__main__":
    shortname_map = get_model_shortname_class_map()

    all_results_mean = {}
    all_results_std = {}

    for dataset_name, ds_class in DATASET_CATALOG.items():
        dataset = ds_class()
        dataset_path = Path(dataset.dataset_path)

        results_mean = {}
        results_std = {}

        results_path = dataset_path / "lp_results"

        if results_path.exists():
            e_paths = sort_embedding_by_model_class(os.listdir(results_path))

            for result_fn in e_paths:
                model = result_fn.split("_")[3]

                if "o250" in result_fn:
                    continue

                if "utronly" in model:
                    continue

                if "rnabert" in model:
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
        # ================
        "#17becf",
        "#bcbd22",
        "#e377c2",
        "#8c564b",
        "#7f7f7f",
        "#1f77b4",
        "#ffbb78",
        "#2ca02c",
        "#d62728"
    ]

    for dataset_name, metrics in all_results_mean.items():
        model_names = ["_".join(k.split("_")[:-1]) for k in metrics.keys()]

        colours = []
        seen_model_class = set()
        c_ind = -1

        # Assumes model_names sorted
        for model_name in model_names:
            model_class = shortname_map[model_name]
            if model_class not in seen_model_class:
                seen_model_class.add(model_class)
                c_ind += 1
            colours.append(CB_color_cycle[c_ind])

        ax = plt.gca()
        f = plt.figure(figsize=(10, 5))

        x_pos = np.arange(len(model_names))
        plt.bar(
            x_pos,
            metrics.values(),
            color=colours,
            yerr=list(all_results_std[dataset_name].values()),
        )

        text_anchor = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
        for i, x in enumerate(x_pos):
            plt.text(
                x - 0.25,
                text_anchor,
                model_names[i],
                rotation=90,
                c="white",
                fontsize=12
            )

        if dataset_name in ["prot-loc", "go-mf"]:
            plt.ylabel("AUPRC")
        else:
            plt.ylabel("Pearson's R")

        plt.title("Linear Probe " + dataset_name)
        ax.axes.get_xaxis().set_visible(False)
        plt.xticks([])
        f.tight_layout()

        plt.savefig("./output/lp_results_{}.png".format(dataset_name))
        plt.close()
