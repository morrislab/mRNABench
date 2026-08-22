import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Compute regression error and correlation metrics."""
    return {
        "mse": float(np.mean((predictions - targets) ** 2)),
        "r": float(pearsonr(predictions, targets).statistic),
        "p": float(spearmanr(predictions, targets).statistic),
    }


def classification_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    missing_class_nan: bool = False,
) -> dict[str, float]:
    """Compute binary or multiclass metrics from class probabilities."""
    labels = classes[np.argmax(scores, axis=1)]
    metrics = {
        "mcc": float(matthews_corrcoef(targets, labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, labels)
        ),
    }
    if scores.shape[1] == 2:
        if missing_class_nan and len(np.unique(targets)) < 2:
            metrics.update({"auroc": float("nan"), "auprc": float("nan")})
        else:
            metrics.update({
                "auroc": float(roc_auc_score(targets, scores[:, 1])),
                "auprc": float(average_precision_score(
                    targets,
                    scores[:, 1],
                    pos_label=classes[1],
                )),
            })
        return metrics

    binary_targets = label_binarize(targets, classes=classes)
    probability_metrics = {
        "accuracy": float(accuracy_score(targets, labels)),
        "f1_macro": float(f1_score(targets, labels, average="macro")),
        "auroc_micro": float(roc_auc_score(
            binary_targets, scores, average="micro", multi_class="ovr"
        )),
        "auprc_micro": float(average_precision_score(
            binary_targets, scores, average="micro"
        )),
    }
    if missing_class_nan and set(np.unique(targets)) != set(classes):
        probability_metrics.update({
            "auroc_macro": float("nan"),
            "auprc_macro": float("nan"),
        })
    else:
        probability_metrics.update({
            "auroc_macro": float(roc_auc_score(
                binary_targets, scores, average="macro", multi_class="ovr"
            )),
            "auprc_macro": float(average_precision_score(
                binary_targets, scores, average="macro"
            )),
        })
    metrics.update(probability_metrics)
    return metrics


def multilabel_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    missing_class_nan: bool = False,
) -> dict[str, float]:
    """Compute micro and macro metrics from multilabel probabilities."""
    metrics = {
        "mcc_micro": float(matthews_corrcoef(
            targets.ravel(),
            (scores >= 0.5).astype(int).ravel(),
        )),
    }
    for average in ("micro", "macro"):
        missing_class = (
            average == "micro" and len(np.unique(targets)) < 2
        ) or (
            average == "macro" and any(
                len(np.unique(targets[:, idx])) < 2
                for idx in range(targets.shape[1])
            )
        )
        if missing_class_nan and missing_class:
            metrics["auroc_{}".format(average)] = float("nan")
            metrics["auprc_{}".format(average)] = float("nan")
            continue
        metrics["auroc_{}".format(average)] = float(roc_auc_score(
            targets, scores, average=average
        ))
        metrics["auprc_{}".format(average)] = float(
            average_precision_score(targets, scores, average=average)
        )
    return metrics
