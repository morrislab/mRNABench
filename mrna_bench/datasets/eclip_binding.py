import pandas as pd

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

eCLIP_K562_TOP_RBPS_LIST = [
    'YBX3', 'UCHL5', 'ZNF622', 'DDX3X', 'LIN28B', 'PUM2', 'PABPC4', 'DDX24',
    'IGF2BP1', 'IGF2BP2', 'RBM15', 'FAM120A', 'PUM1', 'SND1', 'DDX6', 'METAP2',
    'FXR2', 'PCBP1', 'TIA1', 'FMR1'
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

eCLIP_HepG2_TOP_RBPS_LIST = [
    'PPIG', 'DDX3X', 'LARP4', 'LIN28B', 'G3BP1', 'NCBP2', 'IGF2BP1', 'AKAP1',
    'PCBP2', 'PABPN1', 'SND1', 'UCHL5', 'DDX55', 'FXR2', 'EIF3H', 'IGF2BP3',
    'SRSF1', 'HLTF', 'LSM11', 'PRPF4'
]

eCLIP_K562_RBPS_LIST = [
    'target_' + col for col in eCLIP_K562_RBPS_LIST
]
eCLIP_HepG2_RBPS_LIST = [
    'target_' + col for col in eCLIP_HepG2_RBPS_LIST
]
eCLIP_K562_TOP_RBPS_LIST = [
    'target_' + col for col in eCLIP_K562_TOP_RBPS_LIST
]
eCLIP_HepG2_TOP_RBPS_LIST = [
    'target_' + col for col in eCLIP_HepG2_TOP_RBPS_LIST
]


class eCLIPBinding(BenchmarkDataset):
    """eCLIP RBP Binding Dataset."""

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize eCLIPBinding dataset.

        Args:
            dataset_name: Dataset name formatted eclip-binding-{exp_name}
                where exp_name is in: {"k562", "hepg2"}.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: Hugging Face URL for dataset.
        """
        if type(self) is eCLIPBinding:
            raise TypeError("eCLIPBinding is an abstract class.")

        super().__init__(
            dataset_name=dataset_name,
            species="human",
            force_redownload=force_redownload,
            hf_url=hf_url
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        raise NotImplementedError("eCLIP binding from raw under construction.")


class eCLIPBindingK562(eCLIPBinding):
    """Concrete class for K562 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize K562 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.all_cols = eCLIP_K562_RBPS_LIST

        super().__init__(
            "eclip-binding-k562",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "eclip/resolve/main/eclip-k562.parquet"
            )
        )


class eCLIPBindingHepG2(eCLIPBinding):
    """Concrete class for HepG2 cell line experiments."""

    def __init__(self, force_redownload=False):
        """Initialize HepG2 dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        self.all_cols = eCLIP_HepG2_RBPS_LIST

        super().__init__(
            "eclip-binding-hepg2",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "eclip/resolve/main/eclip-hepg2.parquet"
            )
        )
