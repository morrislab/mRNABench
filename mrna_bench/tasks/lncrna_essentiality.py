import pandas as pd

from mrna_bench.tasks.benchmark_dataset import BenchmarkDataset


LNCRNA_URL = "/home/shir2/mRNABench/data/HAP1_essentiality_data.tsv"


class LNCRNAEssentiality(BenchmarkDataset):
    """Long Non-Coding RNA Essentiality Dataset."""
    def __init__(self):
        super().__init__(
            dataset_name="lncRNA Essentiality",
            short_name="lncrna-ess",
            description="TODO",
            raw_data_src_path=LNCRNA_URL
        )

    def process_raw_data(self) -> pd.DataFrame:
        data_df = pd.read_csv(self.raw_data_path, delimiter="\t")

        data_df = data_df[data_df["type"] == "lncRNA"]
        data_df.reset_index(inplace=True, drop=True)
        return data_df
