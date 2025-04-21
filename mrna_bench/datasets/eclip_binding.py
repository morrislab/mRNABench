import json
import numpy as np
import pandas as pd
import os

from pathlib import Path
from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset

eCLIP_K562_RBPS_LIST = [
    'AATF', 'ABCF1', 'AKAP1', 'APOBEC3C', 'AQR', 'BUD13', 'CPEB4', 'CPSF6',
    'CSTF2T', 'DDX21', 'DDX24', 'DDX3X', 'DDX42', 'DDX51', 'DDX52', 'DDX55',
    'DDX6', 'DGCR8', 'DHX30', 'DROSHA', 'EFTUD2', 'EIF3G', 'EIF4G2', 'EWSR1',
    'EXOSC5', 'FAM120A', 'FASTKD2', 'FMR1', 'FXR1', 'FXR2', 'GEMIN5', 'GNL3',
    'GPKOW', 'HLTF', 'HNRNPA1', 'HNRNPC', 'HNRNPL', 'HNRNPM', 'HNRNPU',
    'HNRNPUL1', 'IGF2BP1', 'IGF2BP2', 'ILF3', 'KHDRBS1', 'KHSRP', 'LARP4',
    'LARP7', 'LIN28B', 'MATR3', 'METAP2', 'NCBP2', 'NOLC1', 'NONO', 'NSUN2',
    'PABPC4', 'PCBP1', 'PPIL4', 'PRPF8', 'PTBP1', 'PUM1', 'PUM2', 'PUS1',
    'QKI', 'RBM15', 'RBM22', 'RPS11', 'SAFB', 'SAFB2', 'SBDS', 'SERBP1',
    'SF3B1', 'SF3B4', 'SLBP', 'SLTM', 'SMNDC1', 'SND1', 'SRSF1', 'SRSF7',
    'SSB', 'SUPV3L1', 'TAF15', 'TARDBP', 'TBRG4', 'TIA1', 'TRA2A', 'TROVE2',
    'U2AF1', 'U2AF2', 'UCHL5', 'UTP18', 'UTP3', 'WDR3', 'WDR43', 'YBX3',
    'YWHAG', 'ZC3H11A', 'ZNF622', 'ZRANB2'
]

eCLIP_HepG2_RBPS_LIST = [
    'AKAP1', 'AQR', 'BCCIP', 'BUD13', 'CDC40', 'CSTF2', 'CSTF2T', 'DDX3X',
    'DDX52', 'DDX55', 'DDX6', 'DGCR8', 'DHX30', 'DKC1', 'DROSHA', 'EFTUD2',
    'EIF3D', 'EIF3H', 'EXOSC5', 'FAM120A', 'FASTKD2', 'FKBP4', 'FXR2', 'G3BP1',
    'GRSF1', 'HLTF', 'HNRNPA1', 'HNRNPC', 'HNRNPL', 'HNRNPM', 'HNRNPU',
    'HNRNPUL1', 'IGF2BP1', 'IGF2BP3', 'ILF3', 'KHSRP', 'LARP4', 'LARP7',
    'LIN28B', 'LSM11', 'MATR3', 'NCBP2', 'NIP7', 'NOL12', 'NOLC1', 'PABPN1',
    'PCBP1', 'PCBP2', 'PPIG', 'PRPF4', 'PRPF8', 'PTBP1', 'QKI', 'RBM15',
    'RBM22', 'RBM5', 'SAFB', 'SF3A3', 'SF3B4', 'SLTM', 'SMNDC1', 'SND1',
    'SRSF1', 'SRSF7', 'SRSF9', 'SSB', 'STAU2', 'SUGP2', 'SUPV3L1', 'TAF15',
    'TBRG4', 'TIA1', 'TIAL1', 'TRA2A', 'TROVE2', 'U2AF1', 'U2AF2', 'UCHL5',
    'UTP18', 'WDR43', 'XPO5', 'YBX3', 'ZC3H11A'
]

eCLIP_K562_RBPS_LIST = ['target_' + col for col in eCLIP_K562_RBPS_LIST]
eCLIP_HepG2_RBPS_LIST = ['target_' + col for col in eCLIP_HepG2_RBPS_LIST]


class eCLIPBinding(BenchmarkDataset):
    """eCLIP RBP Binding Dataset."""

    ECLIP_URL = "/data1/morrisq/dalalt1/Orthrus/processed_data/eCLIP/combined_eCLIP_binding.tsv" # noqa

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        **kwargs # noqa
    ):
        """Initialize eCLIPBinding dataset.

        Args:
            dataset_name: Dataset name formatted eclip-binding-{exp_name}
                where exp_name is in: {
                    "k562",
                    "hepg2",
                }.
            force_redownload: Force raw data download even if pre-existing.
        """
        if type(self) is eCLIPBinding:
            raise TypeError("eCLIPBinding is an abstract class.")

        self.isoform_resolved = kwargs.get("isoform_resolved", True)
        self.target_col = kwargs["target_col"]

        valid_targets = ["k562", "hepg2"]
        self.exp_target = next(
            (target for target in valid_targets if target in dataset_name),
            None
        )

        err_msg = f"Invalid experiment target in dataset name: {dataset_name}"
        assert self.exp_target is not None, err_msg

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

        # cds, and splice == strings of lists, convert -> lists -> numpy arrays
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

        keep_cols = [
            "gene",
            "gene_id",
            "transcript_id",
            "isoform_resolved",
            "sequence",
            "cds",
            "splice"
        ]

        df = df[keep_cols + [self.target_col]]

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
        raw_file_name = Path(self.ECLIP_URL).name
        raw_data_path = "{}/{}_{}".format(
            self.raw_data_dir,
            self.exp_target.upper(),
            raw_file_name
        )

        if not os.path.exists(raw_data_path):
            df = pd.read_csv(self.ECLIP_URL, sep="\t")

            # keep the rows that are relevant to this dataset
            df = df[df["cell_line"] == self.exp_target.upper()]
            df = df.reset_index(drop=True)

            # keep the columns that are relevant to this dataset
            keep_cols = [
                "gene",
                "gene_id",
                "transcript_id",
                "isoform_resolved",
                "sequence",
                "cds",
                "splice"
            ]
            df = df[keep_cols + self.all_cols]

            df.to_csv(raw_data_path, sep="\t", index=False)

            self.first_download = True

        self.raw_data_path = raw_data_path


class eCLIPBindingK562(eCLIPBinding):
    """Concrete class for K562 cell line experiments."""

    def __init__(
        self,
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize K562 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.all_cols = eCLIP_K562_RBPS_LIST

        super().__init__("eclip-binding-k562", force_redownload)


class eCLIPBindingHepG2(eCLIPBinding):
    """Concrete class for HepG2 cell line experiments."""

    def __init__(
        self,
        force_redownload=False,
        **kwargs # noqa
    ):
        """Initialize HepG2 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.all_cols = eCLIP_HepG2_RBPS_LIST

        super().__init__("eclip-binding-hepg2", force_redownload)
