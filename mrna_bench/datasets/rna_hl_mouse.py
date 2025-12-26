import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file


RAW_URL = "https://zenodo.org/records/14708163/files/rna_hl_mouse.npz"


class RNAHalfLifeMouse(BenchmarkDataset):
    """RNA Halflife in Mouse Dataset."""

    def __init__(self, force_redownload: bool = False):
        """Initialize RNAHalfLifeMouse dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="rnahl-mouse",
            species="mouse",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "rnahl-saluki-mouse/resolve/main/rnahl-saluki-mouse.parquet"
            )
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        try:
            import genome_kit as gk
            mm_genes = gk.Genome("gencode.vM31").genes
        except ImportError:
            print("GenomeKit is required for raw processing. See README.")
            raise

        print("Downloading raw data...")
        self.raw_data_path = download_file(RAW_URL, self.raw_data_dir)
        data = np.load(self.raw_data_path)
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
            "target": [y for y in data["y"]]
        })

        # Some sequences have no gene name, making the current chromosome
        # lookup impossible. We remove those sequences, but these could be
        # restored by BLASTing the sequences against the genome.
        df = df[df["gene"] != ""]

        chrs = df["gene"].apply(lambda x: mm_genes.first_by_name(x).chromosome)
        chrs = chrs.apply(lambda x: x.replace("chr", ""))

        df.insert(1, "chromosome", chrs)
        df.reset_index(inplace=True, drop=True)

        return df
