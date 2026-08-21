import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.datasets.dataset_utils import (
    create_cds_track,
    create_sequence,
    create_splice_track,
)
from mrna_bench.utils import download_file


HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "rna-lifecycle-ietswaart/resolve/main/"
    "ietswaart_processed.parquet"
)
SOURCE_ARCHIVE = "ietswaart_wf_transcript_tables.tar.gz"
SOURCE_URL = (
    "https://github.com/morrislab/mRNABench/raw/main/"
    f"resources/{SOURCE_ARCHIVE}"
)
SOURCE_SHA256 = (
    "5df456c1241073a3c455ba83690c138f059e91b0643591c64488e4c7ab42a145"
)
LIFECYCLE_QUANTILE = 0.33
COMPARTMENTS = ("chr", "cyto", "poly")

# Preserve the source-table order used to create the published rows.
TABLE_ORDER = {
    "chr": ("K562_chr_1", "K562_chr_2"),
    "cyto": ("K562_cyto_1", "K562_cyto_2"),
    "poly": ("K562_poly_2", "K562_poly_1"),
    "total": ("K562_total_1", "K562_total_2"),
}
REPLICATE_KEYS = [
    "ref_gene_id",
    "ref_id",
    "class_code",
    "num_exons",
    "len",
]
COMPARTMENT_KEYS = ["gene", "transcript_id", "num_exons", "length"]


def _source_archive_is_valid(path: Path) -> bool:
    """Return whether an archive has the expected content checksum."""
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return checksum == SOURCE_SHA256


def _extract_source_tables(
    archive: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Extract the eight transcript tables from the source archive."""
    filenames = {
        alias: f"{alias}_transcripts_table.tsv"
        for aliases in TABLE_ORDER.values()
        for alias in aliases
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(
        str(archive),
        str(output_dir),
        filter="data",
    )
    paths = {
        alias: output_dir / filename
        for alias, filename in filenames.items()
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{SOURCE_ARCHIVE} is missing files: {sorted(missing)}"
        )
    return paths


def _normalized_compartment_coverage(
    table_paths: dict[str, Path],
) -> pd.DataFrame:
    """Merge replicates and calculate per-transcript coverage proportions."""
    compartment_tables = {}
    for compartment, aliases in TABLE_ORDER.items():
        replicates = []
        for alias in aliases:
            table = pd.read_csv(table_paths[alias], sep="\t")
            replicates.append(
                table.loc[table["cov"] > 0].drop(
                    columns=[
                        "qry_id",
                        "parent gene iso num",
                        "sample_id",
                    ]
                )
            )

        merged = replicates[0].merge(
            replicates[1],
            on=REPLICATE_KEYS,
            how="inner",
            sort=False,
        )
        merged = merged.loc[merged["class_code"] == "="].drop(
            columns="class_code"
        )
        merged[f"coverage_{compartment}"] = merged[
            ["cov_x", "cov_y"]
        ].mean(axis=1)
        compartment_tables[compartment] = merged.drop(
            columns=["cov_x", "cov_y"]
        ).rename(
            columns={
                "ref_gene_id": "gene",
                "ref_id": "transcript_id",
                "len": "length",
            }
        )

    combined = compartment_tables["chr"]
    for compartment in ("cyto", "poly", "total"):
        combined = combined.merge(
            compartment_tables[compartment],
            on=COMPARTMENT_KEYS,
            how="outer",
            sort=False,
        )

    coverage_columns = [
        f"coverage_{compartment}" for compartment in COMPARTMENTS
    ]
    combined[coverage_columns] = combined[coverage_columns].fillna(0)
    combined = combined.loc[combined["coverage_total"] > 0].drop(
        columns="coverage_total"
    )
    combined = combined.loc[
        combined[coverage_columns].sum(axis=1) > 0
    ].copy()
    combined[coverage_columns] = combined[coverage_columns].div(
        combined[coverage_columns].sum(axis=1),
        axis=0,
    )
    return combined.reset_index(drop=True)


def _make_lifecycle_targets(coverage: pd.DataFrame) -> np.ndarray:
    """Label the strict lower 33rd percentile in each compartment."""
    columns = [
        f"coverage_{compartment}" for compartment in COMPARTMENTS
    ]
    values = coverage[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Compartment coverage contains non-finite values.")
    cutoffs = np.quantile(values, LIFECYCLE_QUANTILE, axis=0)
    return (values < cutoffs).astype(np.int64)


def _add_sequence_features(
    coverage: pd.DataFrame,
    targets: np.ndarray,
) -> pd.DataFrame:
    """Add GENCODE v47 sequences and structural tracks."""
    try:
        import genome_kit as gk
    except ImportError:
        print(
            "GenomeKit is required for raw processing with GENCODE v47. "
            "Install the mrna-bench dev dependencies."
        )
        raise

    genome = gk.Genome("gencode.v47")
    transcripts = {
        transcript.id: transcript for transcript in genome.transcripts
    }
    missing = sorted(set(coverage["transcript_id"]) - set(transcripts))
    if missing:
        raise ValueError(
            f"{len(missing)} transcript IDs are absent from GENCODE v47: "
            f"{missing[:5]}"
        )

    rows = []
    for transcript_id, target in zip(
        coverage["transcript_id"],
        targets,
        strict=True,
    ):
        transcript = transcripts[transcript_id]
        rows.append(
            {
                "transcript_id": transcript.id,
                "gene": transcript.gene.name,
                "chromosome": transcript.chrom.removeprefix("chr"),
                "sequence": create_sequence(transcript, genome).upper(),
                "cds": create_cds_track(transcript).astype(np.int64),
                "splice": create_splice_track(transcript).astype(
                    np.int64
                ),
                "target": target,
            }
        )
    return pd.DataFrame(rows)


class RNALifecycleIetswaart(BenchmarkDataset):
    """Normalized compartment-coverage labels from human K562 cells."""

    METADATA = DatasetMetadata(
        dataset_name="rna-lifecycle-ietswaart",
        species="human",
        task=["multilabel"],
        target_col=["target"],
        default_split_type="homology",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize the RNA lifecycle dataset.

        Args:
            force_redownload_hf: Force redownload from Hugging Face.
            force_rebuild_raw: Rebuild from the source transcript tables.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Rebuild the published dataset from transcript-level coverages."""
        raw_data_dir = Path(self.raw_data_dir)
        repository_root = Path(__file__).resolve().parents[2]
        archive = repository_root / "resources" / SOURCE_ARCHIVE
        if not _source_archive_is_valid(archive):
            archive = Path(
                download_file(
                    SOURCE_URL,
                    self.raw_data_dir,
                    expected_sha256=SOURCE_SHA256,
                )
            )
        table_paths = _extract_source_tables(archive, raw_data_dir)
        coverage = _normalized_compartment_coverage(table_paths)
        targets = _make_lifecycle_targets(coverage)
        return _add_sequence_features(coverage, targets)
