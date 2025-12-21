import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file


RAW_URL = "https://zenodo.org/records/14708163/files/rna_hl_human.npz"


class RNAHalfLifeHuman(BenchmarkDataset):
    """RNA Halflife in Human Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize RNAHalfLifeHuman dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="rnahl-human",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "rnahl-saluki-human/resolve/main/rnahl-saluki-human.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Download and process raw data from source.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        try:
            import genome_kit as gk
            hg_genes = gk.Genome("gencode.v41").genes
        except ImportError:
            print("GenomeKit is required for raw processing. See README.")
            raise

        print("Downloading raw data...")
        raw_data_path = download_file(RAW_URL, self.raw_data_dir)
        data = np.load(raw_data_path)
        X = data["X"]

        print("Processing raw data...")
        seq_str = ohe_to_str(X[:, :, :4])
        lens = [len(s) for s in seq_str]
        cds = [X[i, :lens[i], 4] for i in range(len(X))]
        splice = [X[i, :lens[i], 5] for i in range(len(X))]

        df = pd.DataFrame({
            "gene": data["genes"],
            "sequence": seq_str,
            "cds": cds,
            "splice": splice,
            "target": [y for y in data["y"]],
        })

        # Some sequences have no gene name, making the current chromosome
        # lookup impossible. We remove those sequences, but these could be
        # restored by BLASTing the sequences against the genome.
        df = df[df["gene"] != ""]

        chrs = df["gene"].apply(lambda x: hg_genes.first_by_name(x).chromosome)
        chrs = chrs.apply(lambda x: x.replace("chr", ""))

        df.insert(1, "chromosome", chrs)
        df.reset_index(inplace=True, drop=True)

        return df
