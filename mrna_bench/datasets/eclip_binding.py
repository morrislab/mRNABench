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


SOURCE_ARCHIVE = "eclip_peakhood_transcript_tables.tar.gz"
SOURCE_URL = (
    "https://github.com/morrislab/mRNABench/raw/main/"
    f"resources/{SOURCE_ARCHIVE}"
)
SOURCE_SHA256 = (
    "8495dc56f3aa2d7d9b52caf6291e2b328781d4d83a2ecfded209dd1919b94df2"
)
SOURCE_FILES = {
    "k562": (
        "K562_transcripts_with_sites.all_tr.tsv",
        "K562_transcripts_with_sites.sel_tr.tsv",
    ),
    "hepg2": (
        "HepG2_transcripts_with_sites.all_tr.tsv",
        "HepG2_transcripts_with_sites.sel_tr.tsv",
    ),
}

ECLIP_K562_RBPS_LIST = [
    "AATF",
    "ABCF1",
    "AKAP1",
    "APOBEC3C",
    "AQR",
    "BUD13",
    "CPEB4",
    "CPSF6",
    "CSTF2T",
    "DDX21",
    "DDX24",
    "DDX3X",
    "DDX42",
    "DDX51",
    "DDX52",
    "DDX55",
    "DDX6",
    "DGCR8",
    "DHX30",
    "DROSHA",
    "EFTUD2",
    "EIF3G",
    "EIF4G2",
    "EWSR1",
    "EXOSC5",
    "FAM120A",
    "FASTKD2",
    "FMR1",
    "FXR1",
    "FXR2",
    "GEMIN5",
    "GNL3",
    "GPKOW",
    "HLTF",
    "HNRNPA1",
    "HNRNPC",
    "HNRNPL",
    "HNRNPM",
    "HNRNPU",
    "HNRNPUL1",
    "IGF2BP1",
    "IGF2BP2",
    "ILF3",
    "KHDRBS1",
    "KHSRP",
    "LARP4",
    "LARP7",
    "LIN28B",
    "MATR3",
    "METAP2",
    "NCBP2",
    "NOLC1",
    "NONO",
    "NSUN2",
    "PABPC4",
    "PCBP1",
    "PPIL4",
    "PRPF8",
    "PTBP1",
    "PUM1",
    "PUM2",
    "PUS1",
    "QKI",
    "RBM15",
    "RBM22",
    "RPS11",
    "SAFB",
    "SAFB2",
    "SBDS",
    "SERBP1",
    "SF3B1",
    "SF3B4",
    "SLBP",
    "SLTM",
    "SMNDC1",
    "SND1",
    "SRSF1",
    "SRSF7",
    "SSB",
    "SUPV3L1",
    "TAF15",
    "TARDBP",
    "TBRG4",
    "TIA1",
    "TRA2A",
    "TROVE2",
    "U2AF1",
    "U2AF2",
    "UCHL5",
    "UTP18",
    "UTP3",
    "WDR3",
    "WDR43",
    "YBX3",
    "YWHAG",
    "ZC3H11A",
    "ZNF622",
    "ZRANB2",
]

ECLIP_K562_TOP_RBPS_LIST = [
    "YBX3",
    "UCHL5",
    "ZNF622",
    "DDX3X",
    "LIN28B",
    "PUM2",
    "PABPC4",
    "DDX24",
    "IGF2BP1",
    "IGF2BP2",
    "RBM15",
    "FAM120A",
    "PUM1",
    "SND1",
    "DDX6",
    "METAP2",
    "FXR2",
    "PCBP1",
    "TIA1",
    "FMR1",
]

ECLIP_HEPG2_RBPS_LIST = [
    "AKAP1",
    "AQR",
    "BCCIP",
    "BUD13",
    "CDC40",
    "CSTF2",
    "CSTF2T",
    "DDX3X",
    "DDX52",
    "DDX55",
    "DDX6",
    "DGCR8",
    "DHX30",
    "DKC1",
    "DROSHA",
    "EFTUD2",
    "EIF3D",
    "EIF3H",
    "EXOSC5",
    "FAM120A",
    "FASTKD2",
    "FKBP4",
    "FXR2",
    "G3BP1",
    "GRSF1",
    "HLTF",
    "HNRNPA1",
    "HNRNPC",
    "HNRNPL",
    "HNRNPM",
    "HNRNPU",
    "HNRNPUL1",
    "IGF2BP1",
    "IGF2BP3",
    "ILF3",
    "KHSRP",
    "LARP4",
    "LARP7",
    "LIN28B",
    "LSM11",
    "MATR3",
    "NCBP2",
    "NIP7",
    "NOL12",
    "NOLC1",
    "PABPN1",
    "PCBP1",
    "PCBP2",
    "PPIG",
    "PRPF4",
    "PRPF8",
    "PTBP1",
    "QKI",
    "RBM15",
    "RBM22",
    "RBM5",
    "SAFB",
    "SF3A3",
    "SF3B4",
    "SLTM",
    "SMNDC1",
    "SND1",
    "SRSF1",
    "SRSF7",
    "SRSF9",
    "SSB",
    "STAU2",
    "SUGP2",
    "SUPV3L1",
    "TAF15",
    "TBRG4",
    "TIA1",
    "TIAL1",
    "TRA2A",
    "TROVE2",
    "U2AF1",
    "U2AF2",
    "UCHL5",
    "UTP18",
    "WDR43",
    "XPO5",
    "YBX3",
    "ZC3H11A",
]

ECLIP_HEPG2_TOP_RBPS_LIST = [
    "PPIG",
    "DDX3X",
    "LARP4",
    "LIN28B",
    "G3BP1",
    "NCBP2",
    "IGF2BP1",
    "AKAP1",
    "PCBP2",
    "PABPN1",
    "SND1",
    "UCHL5",
    "DDX55",
    "FXR2",
    "EIF3H",
    "IGF2BP3",
    "SRSF1",
    "HLTF",
    "LSM11",
    "PRPF4",
]

ECLIP_K562_RBPS_LIST = ["target_" + col for col in ECLIP_K562_RBPS_LIST]
ECLIP_HEPG2_RBPS_LIST = ["target_" + col for col in ECLIP_HEPG2_RBPS_LIST]
ECLIP_K562_TOP_RBPS_LIST = [
    "target_" + col for col in ECLIP_K562_TOP_RBPS_LIST
]
ECLIP_HEPG2_TOP_RBPS_LIST = [
    "target_" + col for col in ECLIP_HEPG2_TOP_RBPS_LIST
]
ECLIP_TARGET_COLUMNS = sorted(
    set(ECLIP_K562_RBPS_LIST + ECLIP_HEPG2_RBPS_LIST)
)


def _source_archive_is_valid(path: Path) -> bool:
    """Return whether the archive has the expected checksum."""
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return checksum == SOURCE_SHA256


def _extract_source_tables(
    archive: Path,
    output_dir: Path,
) -> dict[str, tuple[Path, Path]]:
    """Extract the Peakhood transcript tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(
        str(archive),
        str(output_dir),
        filter="data",
    )
    paths = {
        cell_line: (
            output_dir / all_transcripts,
            output_dir / selected_transcripts,
        )
        for cell_line, (
            all_transcripts,
            selected_transcripts,
        ) in SOURCE_FILES.items()
    }
    missing = [
        path.name
        for cell_paths in paths.values()
        for path in cell_paths
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{SOURCE_ARCHIVE} is missing files: {sorted(missing)}"
        )
    return paths


class eCLIPBinding(BenchmarkDataset):
    """eCLIP RBP Binding Dataset."""

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
        hf_url: str = "",
    ):
        """Initialize eCLIPBinding dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
            hf_url: Hugging Face URL for dataset.
        """
        if type(self) is eCLIPBinding:
            raise TypeError("eCLIPBinding is an abstract class.")

        self.cell_line = self.METADATA.dataset_name.rsplit("-", 1)[-1]
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=hf_url,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Rebuild eCLIP binding data from Peakhood transcript tables."""
        try:
            import genome_kit as gk
        except ImportError:
            print(
                "GenomeKit is required for raw processing with Ensembl v112. "
                "Install the mrna-bench dev dependencies."
            )
            raise

        repository_root = Path(__file__).resolve().parents[2]
        source_archive = repository_root / "resources" / SOURCE_ARCHIVE
        if not _source_archive_is_valid(source_archive):
            source_archive = Path(
                download_file(
                    SOURCE_URL,
                    self.raw_data_dir,
                    expected_sha256=SOURCE_SHA256,
                )
            )
        source_paths = _extract_source_tables(
            source_archive,
            Path(self.raw_data_dir),
        )
        all_path, selected_path = source_paths[self.cell_line]
        all_rows = pd.read_csv(
            all_path,
            sep="\t",
            usecols=["transript_id", "site_ids"],
        )
        selected_ids = set(
            pd.read_csv(
                selected_path,
                sep="\t",
                usecols=["transript_id"],
            )["transript_id"]
        )
        rows = all_rows.loc[
            all_rows["transript_id"].isin(selected_ids)
        ].copy()
        rows["bound_rbps"] = rows["site_ids"].map(
            lambda site_ids: {
                site.split("_", 1)[0]
                for site in site_ids.split(",")
            }
        )

        genome = gk.Genome("ensembl.v112")
        transcript_ids = set(rows["transript_id"])
        transcripts = {
            transcript.id.split(".")[0]: transcript
            for transcript in genome.transcripts
            if transcript.id.split(".")[0] in transcript_ids
        }
        missing = sorted(transcript_ids - set(transcripts))
        if missing:
            raise ValueError(
                f"{len(missing)} transcripts are absent from Ensembl v112: "
                f"{missing[:5]}"
            )

        target_index = {
            column.removeprefix("target_"): index
            for index, column in enumerate(ECLIP_TARGET_COLUMNS)
        }
        targets = np.zeros(
            (len(rows), len(ECLIP_TARGET_COLUMNS)),
            dtype=np.int64,
        )
        records = []
        for output_index, row in enumerate(rows.itertuples(index=False)):
            transcript = transcripts[row.transript_id]
            for rbp in row.bound_rbps:
                targets[output_index, target_index[rbp]] = 1
            records.append({
                "transcript_id": transcript.id,
                "gene": transcript.gene.name,
                "chromosome": transcript.chrom.removeprefix("chr"),
                "sequence": create_sequence(transcript, genome).upper(),
                "cds": create_cds_track(transcript),
                "splice": create_splice_track(transcript),
            })

        return pd.concat(
            [
                pd.DataFrame(records),
                pd.DataFrame(
                    targets,
                    columns=ECLIP_TARGET_COLUMNS,
                ),
            ],
            axis=1,
        )


class eCLIPBindingK562(eCLIPBinding):
    """Concrete class for K562 cell line experiments."""

    METADATA = DatasetMetadata(
        dataset_name="eclip-binding-k562",
        species="human",
        task=["classification"],
        target_col=ECLIP_K562_TOP_RBPS_LIST,
        default_split_type="homology",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize K562 dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "eclip/resolve/main/eclip-k562.parquet"
            ),
        )


class eCLIPBindingHepG2(eCLIPBinding):
    """Concrete class for HepG2 cell line experiments."""

    METADATA = DatasetMetadata(
        dataset_name="eclip-binding-hepg2",
        species="human",
        task=["classification"],
        target_col=ECLIP_HEPG2_TOP_RBPS_LIST,
        default_split_type="homology",
        benchmark_set="core",
        evaluations=("linear_probe",),
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize HepG2 dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "eclip/resolve/main/eclip-hepg2.parquet"
            ),
        )
