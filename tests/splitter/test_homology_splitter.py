from pathlib import Path
from unittest.mock import Mock, patch
import zipfile

import pandas as pd
import pytest

from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.data_splitter.homology_split import (
    HomologySplitter,
    _write_csv,
    build_homology_map,
    train_test_split_homologous,
)


PARALOG_COL = "Human paralogue associated gene name"
IDENTITY_COL = (
    "Paralogue %id. target Human gene identical to query gene"
)


def _paralog_df() -> pd.DataFrame:
    rows = [
        ("A", "B", 40),
        ("B", "A", 40),
        ("C", "D", 35),
        ("D", "C", 35),
        ("E", "F", 36),
        ("H", "I", 41),
        ("I", "H", 41),
        ("H", "I", 20),
    ]
    rows.extend(
        ("G{}".format(i), "G{}".format(i + 1), 50)
        for i in range(7)
    )
    return pd.DataFrame(
        rows,
        columns=["Gene name", PARALOG_COL, IDENTITY_COL],
    )


def test_build_homology_map_reproduces_published_filters():
    """Identity and connectedness filters match the published maps."""
    homology = build_homology_map(_paralog_df())

    assert set(homology["gene_name"]) == {
        "A", "B", "E", "F", "H", "I",
    }
    assert set(homology["n_members"]) == {2}
    assert set(homology["perc_connected"]) == {0.5, 1.0}


def test_build_homology_map_accepts_alternate_threshold():
    """Callers can rebuild groups with alternate thresholds."""
    homology = build_homology_map(
        _paralog_df(),
        similarity_threshold=40,
    )

    assert set(homology["gene_name"]) == {
        "H", "I",
    }


@pytest.mark.parametrize(
    "threshold",
    [-1, 101, float("nan"), "35", True],
)
def test_build_homology_map_rejects_invalid_threshold(threshold):
    """Similarity thresholds must be finite percentages."""
    with pytest.raises(ValueError, match="similarity_threshold"):
        build_homology_map(_paralog_df(), threshold)


def test_build_homology_map_rejects_invalid_identity():
    """Malformed identities cannot silently remove homologous pairs."""
    paralogs = _paralog_df()
    paralogs[IDENTITY_COL] = paralogs[IDENTITY_COL].astype(object)
    paralogs.loc[0, IDENTITY_COL] = "invalid"

    with pytest.raises(ValueError, match="identities must be numeric"):
        build_homology_map(paralogs)


def test_build_homology_map_handles_empty_result():
    """A valid threshold can produce a reusable empty map."""
    homology = build_homology_map(
        _paralog_df(),
        similarity_threshold=100,
    )

    assert homology.empty
    assert list(homology.columns) == [
        "gene_name",
        "gene_group",
        "perc_connected",
        "n_members",
    ]


def test_build_homology_map_merges_chained_conflicts():
    """Accepted transitive merges resolve to one canonical group."""
    edges = [
        ("B", "F"), ("B", "D"), ("B", "C"), ("A", "H"),
        ("D", "F"), ("E", "F"), ("A", "G"), ("D", "H"),
        ("A", "E"), ("A", "D"), ("A", "B"), ("C", "D"),
        ("B", "G"), ("B", "E"),
    ]
    paralogs = pd.DataFrame(
        [(gene, paralog, 90) for gene, paralog in edges],
        columns=["Gene name", PARALOG_COL, IDENTITY_COL],
    )

    homology = build_homology_map(paralogs)

    assert homology["gene_group"].nunique() == 1


def test_unknown_genes_stay_together():
    """Repeated rows for an unmapped gene cannot leak across splits."""
    genes = ["known-a", "unknown", "unknown", "known-b"]
    homology = pd.DataFrame({
        "gene_name": ["known-a", "known-b"],
        "gene_group": [0, 1],
    })

    train, test = train_test_split_homologous(
        genes,
        homology,
        test_size=0.5,
        random_state=4,
    )

    assert ({1, 2}.issubset(train)) != ({1, 2}.issubset(test))


@pytest.mark.parametrize(
    "test_size",
    [-1, 1.1, float("nan"), "0.5", True],
)
def test_split_rejects_invalid_test_size(test_size):
    """Direct split calls reject invalid fractions."""
    with pytest.raises(ValueError, match="test_size"):
        train_test_split_homologous(
            ["A", "B"],
            pd.DataFrame({
                "gene_name": ["A", "B"],
                "gene_group": [0, 1],
            }),
            test_size=test_size,
        )


def test_splitter_rejects_null_genes(tmp_path):
    """Missing gene names cannot be assigned leakage-safe groups."""
    map_dir = tmp_path / "ensembl-110" / "sim-35pct"
    map_dir.mkdir(parents=True)
    pd.DataFrame({
        "gene_name": ["A"],
        "gene_group": [0],
    }).to_csv(map_dir / "hsapiens.csv", index=False)
    splitter = HomologySplitter(
        species="human",
        homology_map_path=str(tmp_path),
    )

    with pytest.raises(ValueError, match="Gene cannot be null"):
        splitter.split_df(
            pd.DataFrame({"gene": ["A", None]}),
            test_size=0.5,
            random_seed=1,
        )


def test_splitter_builds_map_from_source_export(tmp_path):
    """The splitter can generate a map directly from a paralog export."""
    source = tmp_path / "human_paralogs.tsv"
    _paralog_df().to_csv(source, sep="\t", index=False)

    splitter = HomologySplitter(
        species="human",
        homology_map_path=str(tmp_path),
        homology_source_path=str(source),
        similarity_threshold=40,
    )

    assert set(splitter.homology_df["gene_name"]) == {
        "H", "I",
    }
    first_cache = list(
        (tmp_path / "ensembl-110").glob(
            "custom-*/sim-40pct/hsapiens.csv"
        )
    )
    assert len(first_cache) == 1
    cached = HomologySplitter(
        species="human",
        homology_map_path=str(tmp_path),
        homology_source_path=str(source),
        similarity_threshold=40,
    )
    assert cached.homology_df.equals(splitter.homology_df)

    changed = _paralog_df().iloc[:1]
    changed.to_csv(source, sep="\t", index=False)
    changed_splitter = HomologySplitter(
        species="human",
        homology_map_path=str(tmp_path),
        homology_source_path=str(source),
        similarity_threshold=40,
    )
    assert len(list(
        (tmp_path / "ensembl-110").glob(
            "custom-*/sim-40pct/hsapiens.csv"
        )
    )) == 2
    assert not changed_splitter.homology_df.equals(splitter.homology_df)


def test_published_archive_materializes_requested_species_and_is_reused(
    tmp_path,
):
    """The retained archive lazily materializes each requested species."""
    archive = tmp_path / HomologySplitter.HOMO_URL.rsplit("/", 1)[-1]
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr(
            "human_homology_map.csv",
            "gene_name,gene_group\nA,0\nB,0\n",
        )
        zip_ref.writestr(
            "mouse_homology_map.csv",
            "gene_name,gene_group\nC,0\nD,0\n",
        )

    with patch(
        "mrna_bench.data_splitter.homology_split.download_file",
        side_effect=AssertionError("archive should be reused"),
    ):
        human = HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
        )
        mouse_map = (
            tmp_path / "ensembl-110" / "sim-35pct" / "mmusculus.csv"
        )
        assert not mouse_map.exists()
        mouse = HomologySplitter(
            species="mouse",
            homology_map_path=str(tmp_path),
        )

    assert set(human.homology_df["gene_name"]) == {"A", "B"}
    assert set(mouse.homology_df["gene_name"]) == {"C", "D"}
    assert (
        tmp_path / "ensembl-110" / "sim-35pct" / "hsapiens.csv"
    ).exists()
    assert mouse_map.exists()


def test_published_map_does_not_reuse_custom_cache(tmp_path):
    """Custom 35% maps cannot shadow the published map."""
    source = tmp_path / "custom.tsv"
    _paralog_df().to_csv(source, sep="\t", index=False)
    HomologySplitter(
        species="human",
        homology_map_path=str(tmp_path),
        homology_source_path=str(source),
        similarity_threshold=35,
    )

    archive = tmp_path / "published.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr(
            "human_homology_map.csv",
            "gene_name,gene_group\npublished,0\n",
        )
    with patch(
        "mrna_bench.data_splitter.homology_split.download_file",
        return_value=str(archive),
    ):
        splitter = HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
        )

    assert splitter.homology_df["gene_name"].tolist() == ["published"]


def test_corrupt_published_archive_is_replaced(tmp_path):
    """A partial retained download does not poison future runs."""
    archive = tmp_path / HomologySplitter.HOMO_URL.rsplit("/", 1)[-1]
    archive.write_bytes(b"partial download")
    replacement = tmp_path / "replacement.zip"
    with zipfile.ZipFile(replacement, "w") as zip_ref:
        zip_ref.writestr(
            "human_homology_map.csv",
            "gene_name,gene_group\nA,0\n",
        )

    with patch(
        "mrna_bench.data_splitter.homology_split.download_file",
        return_value=str(replacement),
    ) as download:
        splitter = HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
        )

    download.assert_called_once()
    assert splitter.homology_df["gene_name"].tolist() == ["A"]
    assert zipfile.is_zipfile(archive)


def test_splitter_rejects_conflicting_cached_map(tmp_path):
    """A cached gene cannot silently map to multiple groups."""
    map_dir = tmp_path / "ensembl-110" / "sim-35pct"
    map_dir.mkdir(parents=True)
    pd.DataFrame({
        "gene_name": ["A", "A"],
        "gene_group": [0, 1],
    }).to_csv(map_dir / "hsapiens.csv", index=False)

    with pytest.raises(ValueError, match="multiple homology groups"):
        HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
        )


def test_split_rejects_null_homology_group():
    """Malformed maps cannot silently create unstable group keys."""
    with pytest.raises(ValueError, match="cannot be null"):
        train_test_split_homologous(
            ["A"],
            pd.DataFrame({
                "gene_name": ["A"],
                "gene_group": [None],
            }),
        )


def test_csv_cache_write_is_atomic(tmp_path):
    """A failed write leaves the prior cache intact."""
    path = tmp_path / "map.csv"
    path.write_text("old")
    dataframe = _paralog_df()

    with patch.object(
        dataframe,
        "to_csv",
        side_effect=RuntimeError("write failed"),
    ):
        with pytest.raises(RuntimeError, match="write failed"):
            _write_csv(dataframe, path)

    assert path.read_text() == "old"


def test_compara_generation_caches_by_species_version_and_force(tmp_path):
    """Compara sources are reused unless release, species, or force changes."""
    with patch(
        "mrna_bench.data_splitter.homology_split."
        "download_compara_paralogs",
        return_value=_paralog_df(),
    ) as download:
        HomologySplitter(
            species="xenopus_tropicalis",
            homology_map_path=str(tmp_path),
        )
        HomologySplitter(
            species="xenopus_tropicalis",
            homology_map_path=str(tmp_path),
            similarity_threshold=39,
        )
        HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
            ensembl_version=111,
        )
        HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
            similarity_threshold=40,
        )
        HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
            similarity_threshold=40,
            force_redownload=True,
        )

    assert [entry.args for entry in download.call_args_list] == [
        ("xenopus_tropicalis", 110),
        ("human", 111),
        ("human", 110),
        ("human", 110),
    ]
    assert (
        tmp_path / "ensembl-110" / "generated"
        / "xenopus_tropicalis.tsv"
    ).exists()
    assert (
        tmp_path / "ensembl-111" / "generated"
        / "sim-35pct" / "hsapiens.csv"
    ).exists()
    assert (
        tmp_path / "ensembl-110" / "generated" / "hsapiens.tsv"
    ).exists()


def test_dataset_get_splits_inherits_species_with_kwargs():
    """Dataset-level tuning keeps the metadata species."""
    dataset = Mock()
    dataset.metadata.species = "human"
    dataset.metadata.default_split_type = "homology"
    dataset.data_df = pd.DataFrame({"gene": ["A", "B"]})
    splitter = Mock()
    splitter.get_all_splits_df.return_value = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    splitter_class = Mock(return_value=splitter)
    with patch.dict(
        "mrna_bench.data_splitter.split_catalog.SPLIT_CATALOG",
        {"homology": splitter_class},
    ):
        BenchmarkDataset.get_splits(
            dataset,
            (0.7, 0.15, 0.15),
            split_kwargs={"similarity_threshold": 40},
        )

    splitter_class.assert_called_once_with(
        similarity_threshold=40,
        species="human",
    )


def test_splitter_rejects_multiple_species(tmp_path):
    """Homology maps are built for one dataset species at a time."""
    with pytest.raises(TypeError, match="species must be a string"):
        HomologySplitter(
            species=["human", "mouse"],
            homology_map_path=str(tmp_path),
        )


@pytest.mark.parametrize("version", [0, 110.5, "110", True])
def test_splitter_rejects_invalid_ensembl_version(tmp_path, version):
    """Cache paths cannot silently disagree with the queried release."""
    with pytest.raises(ValueError, match="ensembl_version"):
        HomologySplitter(
            species="human",
            homology_map_path=str(tmp_path),
            similarity_threshold=40,
            ensembl_version=version,
        )
