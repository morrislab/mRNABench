import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata
)
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file


PL_URL = "https://zenodo.org/records/14708163/files/protein_localization_dataset.npz"  # noqa: E501
HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "protein-localization/resolve/main/prot-loc.parquet"
)


class ProteinLocalization(BenchmarkDataset):
    """Protein Subcellular Localization Dataset."""

    METADATA = DatasetMetadata(
        dataset_name="prot-loc",
        species="human",
        task=["multilabel"],
        target_col=["target"],
        default_split_type="homology",
        benchmark_set="extended"
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False
    ):
        """Initialize ProteinLocalization dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        try:
            import genome_kit as gk
            hg_genes = gk.Genome("gencode.v41").genes
        except ImportError:
            print("GenomeKit is required for raw processing. See README.")
            raise

        print("Downloading raw data...")
        self.raw_data_path = download_file(PL_URL, self.raw_data_dir)
        data = np.load(self.raw_data_path)
        X = data["X"]

        print("Processing raw data...")
        seq_str = ohe_to_str(X[:, :, :4])
        lens = [len(s) for s in seq_str]
        cds = [X[i, :lens[i], 4] for i in range(len(X))]
        splice = [X[i, :lens[i], 5] for i in range(len(X))]

        chrs = []
        for gene in data["genes"]:
            transcript_chr = hg_genes.first_by_name(gene).chromosome
            transcript_chr = transcript_chr.replace("chr", "")
            chrs.append(transcript_chr)

        df = pd.DataFrame({
            "gene": data["genes"],
            "chromosome": chrs,
            "sequence": seq_str,
            "cds": cds,
            "splice": splice,
            "target": [y for y in data["y"]],
        })

        return df
