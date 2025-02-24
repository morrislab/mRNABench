import pickle
import json
import shutil
import numpy as np
import pandas as pd 
import os 

from pathlib import Path
from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


class PCGEssentiality(BenchmarkDataset):
    """Protein Coding Gene Essentiality Dataset."""

    PCG_URL = "/data1/morrisq/dalalt1/Orthrus/processed_data/essentiality/sanjana_data/cell_line_essentiality/expression_fixed/PCG_essentiality.tsv"

    def __init__(self, 
        dataset_name: str,
        target_col : str,
        force_redownload: bool = False,
        isoform_resolved: bool = False,
    ):
        """Initialize PCGEssentiality dataset.

        Args:
            dataset_name: Dataset name formatted pcg-ess-{experiment_name}
                where experiment_name is in: {
                    "hap1",
                    "hek293ft",
                    "k562",
                    "mda_mb_231",
                    "thp1",
                    "shared"
                }.
            force_redownload: Force raw data download even if pre-existing.
        """
        if type(self) is PCGEssentiality:
            raise TypeError("PCGEssentiality is an abstract class.")

        self.target_col = target_col
        self.isoform_resolved = isoform_resolved

        self.exp_target = dataset_name.split("-")[-1]
        assert self.exp_target in ["hap1", "hek293ft", "k562", "mda_mb_231", "thp1", "shared"]

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
        raw_file_name = Path(self.PCG_URL).name
        raw_data_path = self.raw_data_dir + "/" + raw_file_name

        if not os.path.exists(raw_data_path):
            shutil.copy(self.PCG_URL, raw_data_path)
            self.first_download = True

        self.raw_data_path = raw_data_path
    
class PCGEssHAP1(PCGEssentiality):
    """Concrete class for HAP1 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize HAP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs["isoform_resolved"]
        self.target_col = kwargs["target_col"]

        super().__init__(dataset_name="pcg-ess-hap1", 
            target_col=self.target_col, 
            force_redownload=force_redownload, 
            isoform_resolved=self.isoform_resolved, 
        )


class PCGEssHEK293FT(PCGEssentiality):
    """Concrete class for HEK293FT cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize HEK293FT dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs["isoform_resolved"]
        self.target_col = kwargs["target_col"]

        super().__init__(dataset_name="pcg-ess-hek293ft", 
            target_col=self.target_col, 
            force_redownload=force_redownload, 
            isoform_resolved=self.isoform_resolved, 
        )


class PCGEssK562(PCGEssentiality):
    """Concrete class for K562 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize K562 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs["isoform_resolved"]
        self.target_col = kwargs["target_col"]

        super().__init__(dataset_name="pcg-ess-k562", 
            target_col=self.target_col, 
            force_redownload=force_redownload, 
            isoform_resolved=self.isoform_resolved, 
        )


class PCGEssMDA_MB_231(PCGEssentiality):
    """Concrete class for MDA_MB_231 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize MDA_MB_231 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs["isoform_resolved"]
        self.target_col = kwargs["target_col"]

        super().__init__(dataset_name="pcg-ess-mda_mb_231", 
            target_col=self.target_col, 
            force_redownload=force_redownload, 
            isoform_resolved=self.isoform_resolved, 
        )


class PCGEssTHP1(PCGEssentiality):
    """Concrete class for THP1 cell line experiments."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize THP1 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs["isoform_resolved"]
        self.target_col = kwargs["target_col"]

        super().__init__(dataset_name="pcg-ess-thp1", 
            target_col=self.target_col, 
            force_redownload=force_redownload, 
            isoform_resolved=self.isoform_resolved, 
        )


class PCGEssShared(PCGEssentiality):
    """Concrete class for Shared cell line essentiality."""

    def __init__(self, 
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize Shared dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.isoform_resolved = kwargs["isoform_resolved"]
        self.target_col = kwargs["target_col"]

        super().__init__(dataset_name="pcg-ess-shared", 
            target_col=self.target_col, 
            force_redownload=force_redownload, 
            isoform_resolved=self.isoform_resolved, 
        )