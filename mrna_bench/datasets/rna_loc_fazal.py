import numpy as np
import pandas as pd

from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.utils import download_file
from mrna_bench.datasets.dataset_utils import (
    create_sequence,
    create_cds_track,
    create_splice_track
)

RNA_LOC_FAZAL_URL = (
    "https://ars.els-cdn.com/content/image/"
    "1-s2.0-S0092867419305550-mmc3.xlsx"
)


class RNALocalizationFazal(BenchmarkDataset):
    """RNA Subcellular Localization Dataset.

    Note: This dataset is not available on Hugging Face Hub as
    it is under an Elsevier User License which prohibits redistribution.
    We cannot redistribute the data, and nor should you. However, we do
    provide functionality to download the data from the original source
    and process it into a format compatible with the rest of mRNAbench.
    """

    def __init__(self, force_redownload: bool = False):
        """Initialize RNALocalizationFazal dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="rna-loc-fazal",
            species="human",
            force_redownload=force_redownload,
            hf_url=None
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
            self.raw_data_dir
        )

        data = pd.read_excel(
            self.raw_data_path,
            sheet_name='Gene lists and orphans'
        )

        data_cols = [
            col for col in data.columns if 'log2FC' in col
        ]

        data = data[['Ensembl_Gene', 'Common_Gene'] + data_cols]

        # reverse log2FC and get proportion in compartments
        data[data_cols] = data[data_cols].apply(np.exp2)
        data[data_cols] = data[data_cols].apply(
            lambda x: x / x.sum(), axis=1
        )

        # create target array
        data['target'] = data[data_cols].apply(lambda x: np.array(x), axis=1)

        data.drop(columns=data_cols, inplace=True)

        id_to_gene = {g.id.split('.')[0]: g for g in genome.genes}

        df_data = []

        for index, row in data.iterrows():
            gene = id_to_gene.get(row['Ensembl_Gene'])

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

            df_data.append({
                "transcript_id": principal.id,
                "gene": gene.name,
                "chromosome": principal.chrom.strip("chr"),
                "sequence": seq,
                "cds": cds,
                "splice": splice,
                "target": (row['target'] >= 0.125).astype(int),
            })

        df = pd.DataFrame(df_data)

        return df
