import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset


LNCRNA_URL = "/home/shir2/mRNABench/data/HAP1_essentiality_data.tsv"


class LNCRNAEssentiality(BenchmarkDataset):
    """Long Non-Coding RNA Essentiality Dataset."""

    def __init__(self, force_redownload: bool = False):
        super().__init__(
            dataset_name="lncrna-ess",
            species=["human"],
            raw_data_src_path=LNCRNA_URL,
            force_redownload=force_redownload
        )

    def process_raw_data(self) -> pd.DataFrame:
        data_df = pd.read_csv(self.raw_data_path, delimiter="\t")

        data_df = data_df[data_df["type"] == "lncRNA"]
        data_df.reset_index(inplace=True, drop=True)
        return data_df
