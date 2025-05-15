import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file

MRLS_URL = "https://zenodo.org/records/14708163/files/mrl_isoform_resolved.npz"


class MRLSugimoto(BenchmarkDataset):
    """Mean Ribosome Load Dataset from Sugimoto et al. 2022."""

    def __init__(self, force_redownload: bool = False):
        """Initialize MRLSugimoto dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="mrl-sugimoto",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mrl-sugimoto/resolve/main/mrl-sugimoto.parquet"
            )
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
        self.raw_data_path = download_file(MRLS_URL, self.raw_data_dir)
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
            "target": [y for y in data["y"]]
        })

        return df
