import json
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

        valid_targets = ["hap1", "hek293ft", "k562", "mda-mb-231", "thp1", "shared"]
        self.exp_target = next((target for target in valid_targets if target in dataset_name), None)
        assert self.exp_target is not None, f"Invalid experiment target in dataset name: {dataset_name}"

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

        return df

    def subset_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subset dataframe to only include relevant columns.
        
        Args:
            df: Dataframe to subset

        Returns:
            Subsetted dataframe.
        """
        
        if self.isoform_resolved:
            df = df[df["isoform_resolved"] == 1].reset_index(drop=True)

        # drop rows with missing target values in the target column
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        df = df[['gene', 'gene_id', 'transcript', 'isoform_resolved', 'sequence', 'cds', 'splice'] + [self.target_col]]

        return df

    def save_processed_df(self, df: pd.DataFrame):
        """Save dataframe to data storage path.

        Args:
            df: Processed dataframe to save.
        """
        df.to_pickle(self.dataset_path + "/data_df.pkl")

        self.data_df = self.subset_df(df)


    def load_processed_df(self) -> bool:
        """Load processed dataframe from data storage path.

        Returns:
            Whether dataframe was successfully loaded to class property.
        """
        try:
            df = pd.read_pickle(self.dataset_path + "/data_df.pkl")

            self.data_df = self.subset_df(df)

        except FileNotFoundError:
            print("Processed data frame not found.")
            return False
        return True

    def get_raw_data(self):
        """Collect the raw data from given local path."""
        raw_file_name = Path(self.LNCRNA_URL).name
        raw_data_path = self.raw_data_dir + "/" + self.exp_target.upper() + "_" + raw_file_name

        if not os.path.exists(raw_data_path):

            df = pd.read_csv(self.LNCRNA_URL, sep="\t")

            # keep the columns that are relevant to this dataset
            label_cols = [col for col in df.columns if self.exp_target.upper() in col]

            df = df[['gene', 'gene_id', 'transcript', 'isoform_resolved', 'sequence', 'cds', 'splice'] + label_cols]

            df.to_csv(raw_data_path, sep="\t", index=False)

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