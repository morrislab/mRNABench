from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genome_kit import Genome

import numpy as np
import pandas as pd

from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import (
    create_sequence,
    create_cds_track,
    create_splice_track
)


class VEPTraitGym(BenchmarkDataset):
    """TraitGym benchmark for variant effect prediction subsetted for mRNA.

    Dataset provides a reference allele and alternative allele, and the
    corresponding pathogenicity as a binary label. The task is to predict this
    label based on the sequence context of the variant in both zero-shot and
    few-shot settings. The dataset is split into Mendelian and complex traits.

    While TraitGym contains variants from all genomic contexts, we subset this
    to only 3'UTR and 5'UTR to focus on mRNA-related variants. Overall, the
    ratio of pathogenic to benign variants is around 1:10. Benign variants
    are matched to have similar sequence context to pathogenic variants.

    The dataset is intended to be evaluated using either zero-shot or
    chromosome-wise cross validation strategy.
    """

    source_hf_url = "hf://datasets/songlab/TraitGym/"
    complex_url = "complex_traits_matched_9/test.parquet"
    mendelian_url = "mendelian_traits_matched_9/test.parquet"

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        hf_url: str | None = None,
    ):
        """Initialize TraitGym dataset.

        Args:
            dataset_name: Dataset name formatted in "vep-traitgym-{data_type}"
                where data_type is in: {"mendelian", "complex"}.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: URL to HF repo the dataset will be downloaded from.
        """
        if type(self) is VEPTraitGym:
            raise TypeError("VEPTraitGym is an abstract class.")

        data_type = dataset_name.split("-")[-1]

        if data_type == "mendelian":
            self.data_url = self.source_hf_url + self.mendelian_url
        elif data_type == "complex":
            self.data_url = self.source_hf_url + self.complex_url
        else:
            raise ValueError("Invalid data type.")

        super().__init__(dataset_name, "human", force_redownload, hf_url)

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process data from TraitGym into six-track dataset."""
        df = self._generate_dataset_from_scratch()
        df.dropna(inplace=True, subset=["ref_cds"])
        df = self._convert_zero_shot_to_lp(df)

        return df

    def _convert_zero_shot_to_lp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert zero-shot dataset to linear probe format.

        Zero-shot dataset contains a single sample with the reference and
        variant sequences. The linear probe dataset contains two samples,
        one for the reference and one for the variant. All targets in the
        reference sample are set to False, and all targets in the variant
        follow the original target.

        Args:
            df: DataFrame containing zero-shot dataset.

        Returns:
            DataFrame containing linear probe dataset.
        """
        suffixes = ["_sequence", "_cds", "_splice"]

        r_df = df.copy().drop(columns=["var" + s for s in suffixes])
        r_df = r_df.rename(columns={"ref" + s: s[1:] for s in suffixes})

        # Set all reference sequences to non-pathogenic
        r_df["label"] = False

        v_df = df.copy().drop(columns=["ref" + s for s in suffixes])
        v_df = v_df.rename(columns={"var" + s: s[1:] for s in suffixes})

        df = pd.concat([r_df, v_df], axis=0)
        df.rename(columns={"label": "target"}, inplace=True)

        # Add description of variant
        df["description"] = df.apply(
            lambda x: f"chr{x['chrom']}:{x['pos']} {x['ref']}:{x['alt']}",
            axis=1
        )

        df = df[[
            "transcript_id",
            "gene",
            "chrom",
            "sequence",
            "cds",
            "splice",
            "target",
            "description",
        ]]

        # Compress track columns to save space
        df["cds"] = df["cds"].apply(lambda x: x.astype(np.int8))
        df["splice"] = df["splice"].apply(lambda x: x.astype(np.int8))

        df.rename(columns={"chrom": "chromosome"}, inplace=True)

        # Drop duplicate negative transcripts
        df = df.drop_duplicates(subset=["transcript_id", "sequence", "target"])
        df.reset_index(drop=True, inplace=True)

        return df

    def _generate_dataset_from_scratch(self):
        """Download and process raw data into six-track.

        This function exists to document the dataset generation process.
        The normal raw data download and processing should be used.

        Takes raw data from TraitGym and filters it for mRNA-related variants.
        Also extracts the transcript sequence context for each variant.
        """
        try:
            import genome_kit as gk
        except ImportError:
            raise ImportError("GenomeKit required generate the dataset.")

        genome = gk.Genome("gencode.v47")

        # Load raw data
        df = pd.read_parquet(self.data_url).copy()

        # Filter for mRNA-related variants
        utr = df[df["match_group"].str.contains("5_prime|3_prime")]

        # Generate sequence context for each variant
        context = utr.apply(
            lambda x: self._generate_sequence_context(
                genome,
                x["chrom"],
                x["pos"],
                "{}:{}".format(x["ref"], x["alt"]),
                "_".join(x["match_group"].split("_")[:2])
            ),
            axis=1
        )

        utr = pd.concat([utr, context], axis=1)

        return utr

    def _generate_sequence_context(
        self,
        ref_genome: "Genome",
        chrom: int,
        pos: int,
        mutation: str,
        region: str
    ) -> pd.Series:
        """Generate sequence for each reference / variant pair.

        Args:
            ref_genome: Reference genome object.
            chrom: Chromosome number.
            pos: Position of the variant.
            mutation: Mutation in the format "REF:ALT".
            region: Region of mRNA variant. Either "5_prime" or "3_prime".

        Returns:
            Pandas Series containing the reference and variant sequence, CDS
            track, splice track, and matched transcript id.
        """
        try:
            import genome_kit as gk
        except ImportError:
            raise ImportError("GenomeKit required generate the dataset.")

        if region not in ["5_prime", "3_prime"]:
            raise ValueError("Region must be either 5_prime or 3_prime.")

        interval = gk.Interval(chrom, "+", pos, pos, "gencode.v47")

        # Find transcripts that overlaps the variant position
        pos_transcripts = ref_genome.transcripts.find_overlapping(interval)
        neg_transcripts = ref_genome.transcripts.find_overlapping(
            interval.as_negative_strand().expand(0, 1)
        )
        transcripts = pos_transcripts + neg_transcripts

        if len(transcripts) == 0:
            print("No overlapping transcript found for variant. Skipping.")
            return pd.Series({})

        # Select highest principality transcript that overlaps
        def appris_sort(t):
            principality = ref_genome.appris_principality(t)
            if principality is None:
                return 100
            return principality

        transcripts = sorted(transcripts, key=appris_sort)

        matched_transcript = None

        for transcript in transcripts:
            if matched_transcript is not None:
                break

            if transcript.strand == "+":
                interval = interval.as_positive_strand()
            else:
                interval = interval.as_negative_strand()

            if region == "5_prime":
                for utr5 in transcript.utr5s:
                    if utr5.contains(interval):
                        matched_transcript = transcript
                        break
            elif region == "3_prime":
                for utr3 in transcript.utr3s:
                    if utr3.contains(interval):
                        matched_transcript = transcript
                        break

        if matched_transcript is None:
            print("No matched transcript found. Skipping.")
            return pd.Series({})

        var_genome = gk.VariantGenome(
            ref_genome,
            ref_genome.variant(
                "chr{}:{}:{}".format(chrom, pos, mutation)
            )
        )

        context = {
            "ref_sequence": create_sequence(matched_transcript, ref_genome),
            "ref_cds": create_cds_track(matched_transcript),
            "ref_splice": create_splice_track(matched_transcript),
            "var_sequence": create_sequence(matched_transcript, var_genome),
            "var_cds": create_cds_track(matched_transcript),
            "var_splice": create_splice_track(matched_transcript),
            "gene": matched_transcript.gene.name,
            "transcript_id": matched_transcript.id
        }

        return pd.Series(context)


class VEPTraitGymMendelian(VEPTraitGym):
    """Mendelian subset of TraitGym benchmark for variant effect prediction."""

    def __init__(self, force_redownload: bool = False):
        """Initialize Mendelian subset of TraitGym dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "vep-traitgym-mendelian",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "vep-traitgym-mrna/resolve/main/vep-traitgym-mendelian.parquet"
            )
        )


class VEPTraitGymComplex(VEPTraitGym):
    """Complex subset of TraitGym benchmark for variant effect prediction."""

    def __init__(self, force_redownload: bool = False):
        """Initialize Complex subset of TraitGym dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "vep-traitgym-complex",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "vep-traitgym-mrna/resolve/main/vep-traitgym-complex.parquet"
            )
        )
