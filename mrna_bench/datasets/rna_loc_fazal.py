import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file
from mrna_bench.datasets.dataset_utils import (
    create_sequence,
    create_cds_track,
    create_splice_track,
)


RNA_LOC_FAZAL_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116008/suppl/"
    "GSE116008_APEX_seq_deseq2_all_data_polyA_hek293t.txt.gz"
)

# APEX-seq construct names in the GEO DESeq2 table, mapped to compartments.
# Order defines the order of the multilabel target vector.
LABEL_LOCATIONS = {
    "NLS": "nucleus",
    "NIK": "nucleolus",
    "LMA": "nuclear_lamina",
    "NUCP": "nuclear_pore",
    "NES": "cytosol",
    "ERM": "er_membrane",
    "OMM": "outer_mitochondrial_membrane",
}

# ER lumen (KDEL) has no significantly enriched transcripts (Fazal et al.),
# so it is not a label, but it is still part of the completeness filter.
# MITO is excluded from the integrated analysis, as in the paper.
FILTER_LOCATIONS = list(LABEL_LOCATIONS) + ["KDEL"]

# Enrichment criteria from Fazal et al. 2019, applied to all compartments.
LOG2FC_THRESHOLD = 0.75
PADJ_THRESHOLD = 0.05


class RNALocalizationFazal(BenchmarkDataset):
    """RNA Subcellular Localization Dataset (APEX-seq, Fazal et al. 2019).

    Labels follow the paper's enrichment definition, applied uniformly to
    every compartment: a gene is enriched in a compartment if its DESeq2
    log2 fold-change over the 18 unlabeled controls is > 0.75 and its
    FDR-adjusted p-value is < 0.05. DESeq2 results are taken from GEO
    (GSE116008); their fold-changes are identical to Table S3 of the paper.

    Genes are kept if they have fold-change and adjusted p-value estimates
    in all eight non-matrix locations and are enriched in at least one
    labelled compartment. The ER lumen is not a label as it has no enriched
    transcripts.
    """

    METADATA = DatasetMetadata(
        dataset_name="rna-loc-fazal",
        species="human",
        task=["multilabel"],
        target_col=["target"],
        default_split_type="homology",
        benchmark_set="core",
        vep=False,
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize RNALocalizationFazal dataset.

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

            genome = gk.Genome("gencode.v29")
        except ImportError:
            print(
                "GenomeKit is required for raw processing"
                " with Gencode v29. See README."
            )
            raise

        print("Downloading raw data...")
        self.raw_data_path = download_file(
            RNA_LOC_FAZAL_URL,
            self.raw_data_dir,
        )

        data = pd.read_csv(self.raw_data_path, sep="\t")

        lfc_cols = [f"{loc}.18C" for loc in FILTER_LOCATIONS]
        padj_cols = [f"{loc}.18C.P" for loc in FILTER_LOCATIONS]
        data = data.dropna(subset=lfc_cols + padj_cols)

        enriched = pd.DataFrame(
            {
                loc: (data[f"{loc}.18C"] > LOG2FC_THRESHOLD) & (
                    data[f"{loc}.18C.P"] < PADJ_THRESHOLD
                )
                for loc in LABEL_LOCATIONS
            }
        )

        data = data[enriched.any(axis=1)]
        enriched = enriched.loc[data.index]

        data = data.assign(target=list(enriched.to_numpy(dtype=int)))
        data = data.sort_values("Ensembl_Gene")

        id_to_gene = {g.id.split(".")[0]: g for g in genome.genes}

        df_data = []

        for index, row in data.iterrows():
            gene: gk.Gene = id_to_gene.get(row["Ensembl_Gene"])

            appris_transcripts = genome.appris_transcripts(gene)

            if len(appris_transcripts) == 0:
                transcripts = gene.transcripts

                # grab the longest transcript if no appris
                principal = max(transcripts, key=lambda t: len(t))
            else:
                principal = appris_transcripts[0]

            cds = create_cds_track(principal)
            splice = create_splice_track(principal)
            seq = create_sequence(principal, genome).upper()

            df_data.append(
                {
                    "transcript_id": principal.id,
                    "gene": gene.name,
                    "chromosome": principal.chrom.strip("chr"),
                    "sequence": seq,
                    "cds": cds,
                    "splice": splice,
                    "target": row["target"],
                }
            )

        df = pd.DataFrame(df_data)

        return df
