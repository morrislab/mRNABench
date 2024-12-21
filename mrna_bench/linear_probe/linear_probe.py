import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import RidgeCV, LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from mrna_bench.data_splitter.data_splitter import DataSplitter
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

from mrna_bench.tasks.benchmark_dataset import BenchmarkDataset
from mrna_bench.tasks.dataset_catalog import DATASET_CATALOG

from mrna_bench.linear_probe.embedder.embedder_utils import get_output_filename


def auprc_mc(true_label: list[int], pred_probs: np.ndarray) -> float:
    """Calculate macro-average multiclass AUPRC.
    
    This method can also be used for multilabel classification, as long as
    the array of true values are structured as: [0, 1, 0, .., 1, 0], etc.

    Args:
        true_label: One hot encoded class labels.
        pred_probs: Predicted class probabilities.
    
    Returns:
        Macro-average multiclass AUPRC.
    """
    average_precision_per_class = []

    for i in range(pred_probs.shape[1]):
        class_lab = (true_label == i).astype(int)
        class_prob = pred_probs[:, i]

        class_auprc = average_precision_score(class_lab, class_prob)

        average_precision_per_class.append(class_auprc)
    return np.mean(average_precision_per_class)


def flatten(arr: list[np.ndarray], n_class: int) -> np.ndarray:
    """Flatten multilabel probability output. Takes probability of positive."""
    return np.hstack([arr[c][:, 1] for c in range(n_class)])


class LinearProbe:
    def __init__(
        self,
        embedding_dir: str,
        dataset_name: str,
        model_short_name: str,
        seq_chunk_overlap: int,
        target_col: str,
        target_task: str,
        split_type: str,
        split_ratios: tuple[float, float, float],
        eval_all_splits: bool
    ):
        self.target_task = target_task

        valid_tasks = ["reg_lin", "reg_ridge", "classif", "multilabel"]
        assert self.target_task in valid_tasks

        self.embedding_dir = embedding_dir
        self.dataset: BenchmarkDataset = DATASET_CATALOG[dataset_name]()
        self.model_name = model_short_name
        self.seq_overlap = seq_chunk_overlap

        self.embeddings_fn = get_output_filename(
            self.embedding_dir,
            self.model_name,
            self.dataset.short_name,
            self.seq_overlap,
        ) + ".npz"

        self.embeddings = np.load(self.embeddings_fn)
        self.data_df = self.dataset.data_df
        self.merge_embeddings()

        self.splitter: DataSplitter = SPLIT_CATALOG[split_type]()
        self.split_ratios = split_ratios
        self.target_col = target_col
        self.eval_all_splits = eval_all_splits

    def merge_embeddings(self):
        """Merge embeddings with benchmark dataframe.

        Assumes that the dataframe rows and embedding order is identical.
        """
        self.data_df["embeddings"] = self.embeddings

    def get_df_splits(self, random_seed: int) -> dict[str, pd.DataFrame]:
        train_df, val_df, test_df = self.splitter.get_all_splits_df(
            self.data_df,
            self.split_ratios,
            random_seed
        )

        return {
            "train_X": np.array(train_df["embeddings"]),
            "val_X": np.array(val_df["embeddings"]),
            "test_X": np.array(test_df["embeddings"]),
            "train_y": train_df[self.target_col],
            "val_y": val_df[self.target_col],
            "test_y": test_df[self.target_col],
        }

    def run_linear_probe(self, random_seed: int = 2541, full: bool = False):
        splits = self.get_df_splits(random_seed)

        models = {
            "reg_lin": LinearRegression(),
            "reg_ridge": RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]),
            "classif": LogisticRegression(max_iter=5000),
            "multilabel": MultiOutputClassifier(
                LogisticRegression(max_iter=5000)
            )
        }
        model = models[self.target_task]

        np.random.seed(random_seed)
        model.fit(splits["train_X"], splits["train_y"])

        if self.target_task in ["reg_lin", "reg_ridge"]:
            metrics = self.eval_regression(model, splits)
        elif self.target_task == "classif":
            metrics = self.eval_classif(model, splits)
        elif self.target_task == "multilabel":
            metrics = self.eval_multilabel(model, splits)
        else:
            raise ValueError("Invalid task.")

        return metrics

    def eval_regression(
        self,
        model: RegressorMixin,
        splits: dict[str, np.ndarray]
    ) -> dict[str, float]:
        outputs = {"val": model.predict(splits["val_X"])}
        if self.eval_all_splits:
            outputs["train"] = model.predict(splits["train_X"])
            outputs["test"] = model.predict(splits["test_X"])

        metrics = {}

        for split_name, split_pred in outputs.items():
            split_y = splits[split_name + "_y"]
            metrics["{}_mse"] = np.mean((split_pred - split_y) ** 2)
            metrics["{}_r"] = pearsonr(split_pred, split_y).statistic

        return metrics

    def eval_classif(
        self,
        model: ClassifierMixin,
        splits: dict[str, np.ndarray]
    ) -> dict[str, float]:
        outputs = {"val": model.predict_proba(splits["val_X"])}
        if self.eval_all_splits:
            outputs["train"] = model.predict(splits["train_X"])
            outputs["test"] = model.predict(splits["test_X"])

        metrics = {}

        for split_name, split_pred in outputs.items():
            split_y = splits[split_name + "_y"]
            metrics["{}_auroc"] = roc_auc_score(split_y, split_pred)
            metrics["{}_auprc"] = average_precision_score(split_y, split_pred)

        return metrics

    def eval_multilabel(
        self,
        model: MultiOutputClassifier,
        splits: dict[str, np.ndarray]
    ) -> dict[str, float]:

        outputs = {"val": model.predict_proba(splits["val_X"])}
        if self.eval_all_splits:
            outputs["train"] = model.predict_proba(splits["train_X"])
            outputs["test"] = model.predict_proba(splits["test_X"])

        metrics = {}

        for split_name, split_pred in outputs.items():
            split_y = splits[split_name + "_y"]
            #metrics["{}_auroc"] = roc_auc_score(split_y, split_pred)
            metrics["{}_auprc"] = auprc_mc(split_y, split_pred)

        return metrics
