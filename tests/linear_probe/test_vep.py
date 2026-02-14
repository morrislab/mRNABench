import pytest

import numpy as np
import pandas as pd

from mrna_bench.linear_probe.vep import compute_vep_deltas


def test_compute_vep_deltas_basic():
    df = pd.DataFrame({
        "transcript_id": ["tx1", "tx1", "tx1"],
        "description": [
            "wild-type",
            "chr1:10 A:T",
            "chr1:20 G:C,missense"
        ],
        "embeddings": [
            np.array([1.0, 1.0]),
            np.array([2.0, 3.0]),
            np.array([4.0, 6.0]),
        ],
        "target": [0, 1, 1],
    })

    out = compute_vep_deltas(df)

    assert len(out) == 2  # wild-type removed

    np.testing.assert_array_equal(
        out.iloc[0]["embeddings"],
        np.array([1.0, 2.0])
    )

    np.testing.assert_array_equal(
        out.iloc[1]["embeddings"],
        np.array([3.0, 5.0])
    )


def test_compute_vep_deltas_missing_wildtype():
    df = pd.DataFrame({
        "transcript_id": ["tx1", "tx1"],
        "description": [
            "chr1:10 A:T",
            "chr1:20 G:C"
        ],
        "embeddings": [
            np.array([2.0, 3.0]),
            np.array([4.0, 6.0]),
        ],
        "target": [1, 1],
    })

    with pytest.raises(ValueError):
        compute_vep_deltas(df)


def test_compute_vep_deltas_multiple_transcripts():
    df = pd.DataFrame({
        "transcript_id": ["tx1", "tx1", "tx2", "tx2"],
        "description": [
            "wild-type",
            "chr1:10 A:T",
            "wild-type",
            "chr2:50 C:G"
        ],
        "embeddings": [
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([10.0, 10.0]),
            np.array([13.0, 14.0]),
        ],
        "target": [0, 1, 0, 1],
    })

    out = compute_vep_deltas(df)

    assert len(out) == 2

    tx1 = out[out["transcript_id"] == "tx1"].iloc[0]
    tx2 = out[out["transcript_id"] == "tx2"].iloc[0]

    np.testing.assert_array_equal(tx1["embeddings"], [1.0, 1.0])
    np.testing.assert_array_equal(tx2["embeddings"], [3.0, 4.0])
