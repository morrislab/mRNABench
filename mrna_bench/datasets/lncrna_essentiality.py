import pickle
import json
import shutil
import numpy as np
import pandas as pd 
import os 

from pathlib import Path
from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class LNCRNAEssentiality(BenchmarkDataset):
    """Long Non-Coding RNA Essentiality Dataset."""

    LNCRNA_URL = "/data1/morrisq/dalalt1/Orthrus/processed_data/essentiality/sanjana_data/cell_line_essentiality/expression_fixed/lncRNA_essentiality.tsv"

    def __init__(self, 
        dataset_name: str,
        force_redownload: bool = False,
        **kwargs # noqa
    ):
        """Initialize LNCRNAEssentiality dataset.

        Args:
            dataset_name: Dataset name formatted lncrna-ess-{experiment_name}
                where experiment_name is in: {
                    "hap1",
                    "hek293ft",
                    "k562",
                    "mda-mb-231",
                    "thp1",
                    "shared"
                }.
            force_redownload: Force raw data download even if pre-existing.
        """
        if type(self) is LNCRNAEssentiality:
            raise TypeError("LNCRNAEssentiality is an abstract class.")

        self.exp_target = dataset_name.split("-")[-1]
        assert self.exp_target in ["hap1", "hek293ft", "k562", "mda-mb-231", "thp1", "shared"]

        super().__init__(
            dataset_name=dataset_name,
            species=["human"],
            force_redownload=force_redownload
        )

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        
        df = pd.read_csv(self.raw_data_path, sep="\t")

        # the cds, and splice == strings of lists, convert -> lists -> numpy arrays
        df["cds"] = df["cds"].apply(lambda x: np.array(json.loads(x)))
        df["splice"] = df["splice"].apply(lambda x: np.array(json.loads(x)))

        if self.isoform_resolved:
            df = df[df["isoform_resolved"] == 1].reset_index(drop=True)

        # drop rows with missing target values in the target column
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        df = df[['gene', 'gene_id', 'transcript', 'isoform_resolved', 'sequence', 'cds', 'splice'] + [self.target_col]]

        return df

    def get_raw_data(self):
        """Collect the raw data from given local path."""
        raw_file_name = Path(self.LNCRNA_URL).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        if not os.path.exists(raw_data_path):
            shutil.copy(self.LNCRNA_URL, raw_data_path)
            self.first_download = True

        self.raw_data_path = raw_data_path

class LNCRNAEssHAP1(LNCRNAEssentiality):
    """Concrete class for HAP1 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize HAP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs.get("target_col", "essential_HAP1")

        super().__init__("lncrna-ess-hap1", force_redownload)

class LNCRNAEssHEK293FT(LNCRNAEssentiality):
    """Concrete class for HEK293FT cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize HEK293FT dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs.get("target_col", "essential_HEK293FT")

        super().__init__("lncrna-ess-hek293ft", force_redownload)

class LNCRNAEssK562(LNCRNAEssentiality):
    """Concrete class for K562 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize K562 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs.get("target_col", "essential_K562")

        super().__init__("lncrna-ess-k562", force_redownload)

class LNCRNAEssMDA_MB_231(LNCRNAEssentiality):
    """Concrete class for MDA-MB-231 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize MDA-MB-231 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs.get("target_col", "essential_MDA-MB-231")

        super().__init__("lncrna-ess-mda-mb-231", force_redownload)

class LNCRNAEssTHP1(LNCRNAEssentiality):
    """Concrete class for THP1 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize THP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs.get("target_col", "essential_THP1")

        super().__init__("lncrna-ess-thp1", force_redownload)

class LNCRNAEssShared(LNCRNAEssentiality):
    """Concrete class for Shared cell line essentiality."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize Shared dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs.get("target_col", "essential_SHARED")

        super().__init__("lncrna-ess-shared", force_redownload)