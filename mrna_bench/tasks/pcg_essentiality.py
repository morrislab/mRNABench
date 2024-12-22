import pandas as pd

from mrna_bench.tasks.benchmark_dataset import BenchmarkDataset


LNCRNA_URL = "/home/shir2/mRNABench/data/HAP1_essentiality_data.tsv"


class PCGEssentiality(BenchmarkDataset):
    """Protein Coding Gene Essentiality Dataset."""
    def __init__(self):
        super().__init__(
            dataset_name="PCG Essentiality",
            short_name="pcg-ess",
            description="TODO",
            species=["human"],
            raw_data_src_path=LNCRNA_URL
        )

    def process_raw_data(self) -> pd.DataFrame:
        data_df = pd.read_csv(self.raw_data_path, delimiter="\t")

        data_df = data_df[data_df["type"] == "pcg"]
        data_df.reset_index(inplace=True, drop=True)
        return data_df
