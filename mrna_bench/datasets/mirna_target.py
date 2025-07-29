import pandas as pd
from tqdm import tqdm

from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.utils import download_file


MIRNA_TARGET_URL = "https://cosbi.ee.ncku.edu.tw/MirTarClash_static/download/human_mRNA_chira_download.csv"
MIRNA_TARGETS = [
    "hsa-miR-92a-3p",
    "hsa-miR-122-5p",
    "hsa-miR-21-5p",
    "hsa-miR-186-5p",
    "hsa-miR-320a-3p",
    "hsa-miR-17-5p",
    "hsa-miR-8485",
    "hsa-miR-484",
    "hsa-miR-20a-5p",
    "hsa-miR-19b-3p",
    "hsa-miR-26a-5p",
    "hsa-miR-615-3p",
    "hsa-miR-16-5p",
    "hsa-miR-194-5p",
    "hsa-miR-30c-5p",
    "hsa-miR-19a-3p",
    "hsa-miR-193b-3p",
    "hsa-miR-93-5p",
    "hsa-miR-103a-3p",
    "hsa-let-7b-5p",
]
MIRNA_TARGETS_WITH_PREFIX = ["target_" + s for s in MIRNA_TARGETS]


class MiRNATarget(BenchmarkDataset):
    """miRNA Target Prediction Dataset.

    Note: This dataset is not available on Hugging Face Hub as
    it is under a specific license which might prohibit redistribution.
    We cannot redistribute the data, and nor should you. However, we do
    provide functionality to download the data from the original source
    and process it into a format compatible with the rest of mRNAbench.
    """

    def __init__(self, force_redownload: bool = False):
        """Initialize MiRNATarget dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="mirna-target",
            species="human",
            force_redownload=force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mirna-target/resolve/main/mirna_preprocessed.parquet"
            ),
        )
        self.all_cols = MIRNA_TARGETS_WITH_PREFIX

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        try:
            import genome_kit as gk
            from mrna_bench.gk_utils import (
                get_top_n_priority_transcripts,
                create_cds_track,
                create_splice_track,
                get_transcript_sequence,
            )

            genome = gk.Genome("gencode.v41")
        except ImportError:
            print(
                "GenomeKit is required for raw processing with Gencode v41. See README."
            )
            raise

        print("Downloading raw data...")
        self.raw_data_path = download_file(MIRNA_TARGET_URL, self.raw_data_dir)

        df = pd.read_csv(self.raw_data_path)
        print(f"Starting with {len(df)} transcripts")

        # Load genome and create gene mapping
        genome_ids = {x.id.split(".")[0] for x in genome.genes}
        id_to_gene = {x.id.split(".")[0]: x for x in genome.genes}

        # Filter dataframe to only include genes that exist in the genome
        df_before_filter = len(df)
        df = df[df["gene ID"].isin(genome_ids)]
        df_after_filter = len(df)
        print(
            f"After filtering to genes in genome: {df_after_filter} "
            f"transcripts (dropped {df_before_filter - df_after_filter})"
        )

        # Group by gene and select top priority transcripts for each gene
        df_filtered = []
        for gene_id, group in tqdm(df.groupby("gene ID"), desc="Filtering transcripts"):
            if gene_id in id_to_gene:
                gene = id_to_gene[gene_id]

                # Get top 5 priority transcripts for this gene
                top_transcripts = get_top_n_priority_transcripts(gene, genome, n=5)

                if not top_transcripts:
                    continue

                # Get transcript IDs from the top transcripts (strip version)
                top_transcript_ids = [t.id.split(".")[0] for t in top_transcripts]

                # Create mapping from transcript ID to transcript object
                transcript_id_to_obj = {t.id.split(".")[0]: t for t in top_transcripts}

                # Filter the group to only include these top transcripts (strip version from mRNA column)
                filtered_group = group[
                    group["mRNA"].str.split(".").str[0].isin(top_transcript_ids)
                ].copy()

                # Add transcript object column to the filtered group
                if not filtered_group.empty:
                    filtered_group["transcript_obj"] = (
                        filtered_group["mRNA"]
                        .str.split(".")
                        .str[0]
                        .map(transcript_id_to_obj)
                    )
                    df_filtered.append(filtered_group)

        # Combine all filtered groups
        if df_filtered:
            df_new = pd.concat(df_filtered, ignore_index=True)
        else:
            df_new = pd.DataFrame()

        df_before_principal = len(df)
        df_after_principal = len(df_new)
        df = df_new
        print(
            f"After filtering to top priority transcripts per gene: "
            f"{df_after_principal} transcripts (dropped "
            f"{df_before_principal - df_after_principal})"
        )

        # Drop genes where gene name is None
        df_before_gene_name_filter = len(df)
        df.dropna(subset=["gene name"], inplace=True)
        df = df[df["gene name"] != "None"]
        df_after_gene_name_filter = len(df)
        print(
            f"After dropping genes with no gene name: "
            f"{df_after_gene_name_filter} transcripts (dropped "
            f"{df_before_gene_name_filter - df_after_gene_name_filter})"
        )

        print(
            f"Dataset filtered to {len(df)} transcripts using top priority transcripts per gene"
        )

        # Split miRNAs column and create binary matrix
        df["miRNAs"] = df["miRNAs"].fillna("-")

        # Create binary columns for each miRNA
        for mirna in tqdm(MIRNA_TARGETS, desc="Creating miRNA binary columns"):
            df["target_" + mirna] = df["miRNAs"].apply(
                lambda x: 1 if mirna in x.split(",") else 0
            )

        df = df.drop("miRNAs", axis=1)

        # Filter to transcripts that have at least one top occurring miRNA
        has_top_mirna = df[MIRNA_TARGETS_WITH_PREFIX].sum(axis=1) > 0
        df_subset = df[has_top_mirna]

        # Select one transcript per gene
        df_subset.dropna(subset=["transcript_obj"], inplace=True)
        df_subset = df_subset.groupby("gene ID").first().reset_index()

        print(
            f"Dataset subset to {len(df_subset)} transcripts with at "
            f"least one of the top 20 miRNAs (excluding most common)"
        )
        print(f"Dataset now has {len(df_subset.columns)} columns")
        print(
            f"Dataset now has one transcript per gene (unique genes:"
            f" {df_subset['gene ID'].nunique()})"
        )

        # Drop rows where 'transcript_obj' is NaN before proceeding
        df_subset.dropna(subset=["transcript_obj"], inplace=True)

        with tqdm(total=len(df_subset), desc="Generating sequences and tracks") as pbar:
            df_subset["transcript_id"] = df_subset["transcript_obj"].apply(
                lambda x: x.id
            )
            pbar.update(0)
            df_subset["cds"] = df_subset["transcript_obj"].apply(
                lambda x: create_cds_track(x)
            )
            pbar.update(0)
            df_subset["sequence"] = df_subset["transcript_obj"].apply(
                lambda x: get_transcript_sequence(x, genome)
            )
            pbar.update(0)
            df_subset["splice"] = df_subset["transcript_obj"].apply(
                lambda x: create_splice_track(x)
            )
            pbar.update(0)
            df_subset["chromosome"] = df_subset["transcript_obj"].apply(
                lambda x: x.chrom
            )
            pbar.update(0)
            df_subset["gene"] = df_subset["transcript_obj"].apply(lambda x: x.gene.name)
            pbar.update(len(df_subset))

        required_cols = [
            "splice",
            "cds",
            "sequence",
            "chromosome",
            "gene",
            "transcript_id",
        ] + MIRNA_TARGETS_WITH_PREFIX

        df_final = df_subset[required_cols].copy()

        return df_final
