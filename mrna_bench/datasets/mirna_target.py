import pandas as pd
from tqdm import tqdm

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file


MIRNA_TARGET_URL = (
    "https://cosbi.ee.ncku.edu.tw/"
    "MirTarClash_static/download/"
    "human_mRNA_chira_download.csv"
)
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

    This dataset contains experimentally validated miRNA target sites on
    human mRNAs. The original dataset has been converted into a binary
    classification task for the top 20 most frequently occurring miRNAs,
    where each target column indicates whether the corresponding miRNA
    targets the mRNA (1) or not (0).

    The raw data is obtained from the MirTarClash database:
    https://cosbi.ee.ncku.edu.tw/MirTarClash/home/

    Note: This dataset is not available on Hugging Face Hub as we cannot find
    its distribution license. We are not redistributing the data, and nor
    should you. However, we do provide functionality to download the data
    from the original source and process it into a format compatible with
    the rest of mRNAbench.
    """

    METADATA = DatasetMetadata(
        dataset_name="mirna-target",
        species="human",
        task=["classification"],
        target_col=MIRNA_TARGETS_WITH_PREFIX,
        default_split_type="homology",
        benchmark_set="core",
        vep=False,
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize MiRNATarget dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        try:
            import genome_kit as gk
            from mrna_bench.datasets.dataset_utils import (
                get_top_n_priority_transcripts,
                create_cds_track,
                create_splice_track,
                create_sequence,
            )

            genome = gk.Genome("gencode.v41")
        except ImportError:
            print(
                "GenomeKit is required for raw processing "
                "with Gencode v41. See README."
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
        for gene_id, group in tqdm(
            df.groupby("gene ID"), desc="Filtering transcripts"
        ):
            if gene_id in id_to_gene:
                gene = id_to_gene[gene_id]

                # Get top 5 priority transcripts for this gene
                top_transcripts = get_top_n_priority_transcripts(
                    gene,
                    genome,
                    n=5,
                )

                if not top_transcripts:
                    continue

                # Get transcript IDs from the top transcripts (strip version)
                top_tx_ids = [t.id.split(".")[0] for t in top_transcripts]

                # Create mapping from transcript ID to transcript object
                transcript_id_to_obj = {
                    t.id.split(".")[0]: t for t in top_transcripts
                }

                # Filter the group to only include
                # these top transcripts (strip version from mRNA column)
                filtered_group = group[
                    group["mRNA"].str.split(".").str[0].isin(top_tx_ids)
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
            f"Dataset filtered to {len(df)} transcripts "
            "using top priority transcripts per gene"
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

        processed_rows = []
        for _, row in tqdm(
            df_subset.iterrows(),
            total=len(df_subset),
            desc="Generating sequences and tracks",
        ):
            transcript_obj = row["transcript_obj"]
            processed_rows.append(
                {
                    "transcript_id": transcript_obj.id,
                    "gene": transcript_obj.gene.name,
                    "chromosome": transcript_obj.chrom.replace("chr", ""),
                    "sequence": create_sequence(transcript_obj, genome),
                    "cds": create_cds_track(transcript_obj),
                    "splice": create_splice_track(transcript_obj),
                    **{col: row[col] for col in MIRNA_TARGETS_WITH_PREFIX},
                }
            )

        df_final = pd.DataFrame(processed_rows)

        for col in MIRNA_TARGETS_WITH_PREFIX:
            df_final[col] = df_final[col].astype("int8")

        return df_final
