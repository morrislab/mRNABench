import numpy as np
import pandas as pd
import warnings


def compute_vep_deltas(
    df: pd.DataFrame,
    embedding_col: str = "embeddings",
    transcript_col: str = "transcript_id",
    description_col: str = "description",
):
    """Compute difference between variant and wild-type embeddings.

    Args:
        df: DataFrame containing embeddings and descriptions.
        embedding_col: Name of the column containing embeddings.
        transcript_col: Name of the column containing transcript IDs.
        description_col: Name of the column containing variant descriptions.
    Returns:
        DataFrame with VEP deltas computed.
    """
    def is_wildtype(desc: str) -> bool:
        return desc.split(",")[-1] == "wild-type"

    df = df.copy()
    df["_is_wt"] = df[description_col].apply(is_wildtype)

    missing_wt = df.groupby(transcript_col)["_is_wt"].sum() == 0
    missing_tx = missing_wt[missing_wt].index.tolist()

    if missing_wt.any():
        warnings.warn(
            "Warning: Missing wild-type for transcripts: "
            f"{', '.join(missing_tx)}. "
            "These transcripts will be skipped.",
            RuntimeWarning,
        )

    df = df[~df[transcript_col].isin(missing_tx)]

    if df["_is_wt"].sum() == 0:
        raise ValueError(
            "No wild-type sequences found in dataframe. "
            "Cannot compute variant effect without wild-type sequences."
        )

    def variant_difference(group):
        wt = group[group["_is_wt"]]
        if wt.empty:
            # This should not happen due to prior filtering
            raise ValueError(
                "No wild-type sequence found for transcript "
                f"{group.name}."
            )

        wt_emb = np.asarray(wt.iloc[0][embedding_col])
        group = group[~group["_is_wt"]].copy()
        group[embedding_col] = group[embedding_col].apply(
            lambda x: np.asarray(x) - wt_emb
        )
        return group

    out = (
        df.groupby(transcript_col, group_keys=True)
        .apply(variant_difference, include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    return out.drop(columns=["_is_wt"])
