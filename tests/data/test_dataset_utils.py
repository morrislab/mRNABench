from mrna_bench.datasets.dataset_utils import str_to_ohe


def test_str_to_ohe():
    """Test str_to_ohe function."""
    out_1 = str_to_ohe("ACGTN")

    assert out_1.shape == (5, 4)
    assert out_1.sum() == 4

    assert out_1[0, 0] == 1
    assert out_1[1, 1] == 1
    assert out_1[2, 2] == 1
    assert out_1[3, 3] == 1
    assert out_1[4].sum() == 0


def test_str_to_ohe_null():
    """Test str_to_ohe function."""
    out_1 = str_to_ohe("NNNN")

    assert out_1.shape == (4, 4)
    assert out_1.sum() == 0
