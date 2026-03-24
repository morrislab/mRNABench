import pandas as pd
from tqdm import tqdm

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file


TRANSLATION_EFFICIENCY_MOUSE_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41587-025-02712-x/"
    "MediaObjects/41587_2025_2712_MOESM4_ESM.xlsx"
)
HF_URL = (
    "https://huggingface.co/datasets/morrislab/"
    "translation-efficiency-mouse/resolve/main/te_mouse.parquet"
)


class TranslationEfficiencyMouse(BenchmarkDataset):
    """Translation Efficiency Prediction Dataset for Mouse."""

    METADATA = DatasetMetadata(
        dataset_name="translation-efficiency-mouse",
        species="mouse",
        task=["regression"],
        target_col=["target"],
        default_split_type="homology",
        benchmark_set="core",
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize TranslationEfficiencyMouse dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_URL,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe."""
        try:
            import genome_kit as gk

            genome = gk.Genome("gencode.vM31")
            from mrna_bench.datasets.dataset_utils import (
                create_cds_track,
                create_splice_track,
                create_sequence,
            )
        except ImportError:
            print(
                "GenomeKit is required for raw processing "
                "with Gencode vM31. See README."
            )
            raise

        print("Downloading raw data...")
        self.raw_data_path = download_file(
            TRANSLATION_EFFICIENCY_MOUSE_URL,
            self.raw_data_dir,
        )

        df = pd.read_excel(self.raw_data_path)
        df.dropna(subset=["tx_id", "mean_te"], inplace=True)

        df["transcript_id_no_version"] = df["tx_id"].str.split(".").str[0]

        id_to_transcript_obj = {
            t.id.split(".")[0]: t
            for t in tqdm(genome.transcripts, desc="Building transcript map")
        }
        df["transcript_obj"] = df["transcript_id_no_version"].map(
            id_to_transcript_obj
        )
        df.dropna(subset=["transcript_obj"], inplace=True)

        df["gene_obj"] = df["transcript_obj"].apply(lambda x: x.gene)
        df["gene_id"] = df["gene_obj"].apply(lambda x: x.id)

        df_subset = (
            df.sort_values(by="mean_te", ascending=False)
            .groupby("gene_id")
            .first()
            .reset_index()
        )

        print(
            f"Filtered to {len(df_subset)} transcripts "
            "with translation efficiency data."
        )

        processed_rows = []
        for _, row in tqdm(
            df_subset.iterrows(),
            total=len(df_subset),
            desc="Generating sequences and tracks",
        ):
            transcript_obj = row["transcript_obj"]
            seq = create_sequence(transcript_obj, genome)
            if not seq:
                continue

            cds_track = create_cds_track(transcript_obj)
            splice_track = create_splice_track(transcript_obj)

            processed_rows.append(
                {
                    "transcript_id": transcript_obj.id,
                    "gene": transcript_obj.gene.name,
                    "chromosome": transcript_obj.chrom.strip("chr"),
                    "sequence": seq.upper(),
                    "cds": cds_track,
                    "splice": splice_track,
                    "target": row["mean_te"].astype("float32"),
                }
            )

        final_df = pd.DataFrame(processed_rows)
        return final_df
