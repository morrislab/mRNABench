from pathlib import Path

import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file


SOURCE_COMMIT = "4aaeb7e97c5ec093c356f0564f96d87887ee9ab7"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/david-a-siegel/"
    f"AU-Rich-Elements/{SOURCE_COMMIT}/"
)
SOURCE_FILES = {
    "jurkat": (
        "sequence_level_data_Jurkat.csv",
        "4b3e1b3452a7753d8308e3eb1c8847c3f2692ff2a08d699ea1e072d52f150f66",
    ),
    "beas2b": (
        "sequence_level_data_Beas2B.csv",
        "39b8cd5fad482e3ddec5b07f6975ff8447b50976c134de966c6e285a520b0ca8",
    ),
}
HF_ROOT = (
    "https://huggingface.co/datasets/morrislab/"
    "rna-stability-siegel/resolve/main/"
)

# Reporter reconstruction published with the Saluki analysis of this dataset:
# https://github.com/vagarwal87/saluki_paper/blob/
# 3aa4e56a19bbbf87ac9f5f0a251098cf749ad6bc/
# Fig6_S7/Siegel_testSet/BTV_construct.txt.gz
BTV_CONSTRUCT_SHA256 = (
    "a237c2b44d94d1e3b007aaea643eea06"
    "bbe320368639ad4983541aea600ae1dc"
)
BTV_5PRIME_UTR = (
    "GTGGTAAACTCGACCTATATAAGCAGAGCTCGTTTAGTGAACCGTCAGATCGCCTGGAGACG"
    "CCATCCACGCTGTTTTGACCTCCATAGAAGACACCGGGACCGATCCAGCCTCCGCGGCCCC"
    "GAATTCCTGCAGCGGCCCTAGCGCTACCGGTCGCCACC"
)
BTV_EGFP_CDS = (
    "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGAC"
    "GGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTAC"
    "GGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACC"
    "CTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAG"
    "CAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTC"
    "TTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTG"
    "GTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCAC"
    "AAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAAC"
    "GGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCC"
    "GACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCAC"
    "TACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTC"
    "CTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAG"
)
BTV_3PRIME_UTR_PREFIX = (
    "CGGCCGGCCGCGTCGACCTAGTTCTAGCTAGCAACTACGCGTAGT"
)
BTV_3PRIME_UTR_SUFFIX = (
    "CGATATCCTTACCTGCAGGGCCTTTAATTAAACTGGCTAGCTTAGTCATGCACCGGTGGAT"
    "CCAGACCACCTCCCCTGCGAGCTAAGCTGGACAGCCAATGACGGGTAAGAGAGTGACATTT"
    "TTCACTAACCTAAGACAGGAGGGCCGTCAGAGCTACTGCCTAATCCAAAGACGGGTAAAAG"
    "TGATAAAAATGTATCACTCCAACCTAAGACAGGCGCAGCTTCCGAGGGATTTGAGATCCAG"
    "ACATGATAAGATACATTGATGAGTTTGGACAAACCAAAACTAGAATGCAGTGAAAAAAATG"
    "CCTTATTTGTGAAATTTGTGATGCTATTGCCTTATTTGTAACCATTATAAGCTGCAATAAA"
    "CAAGTT"
)
BTV_REPORTER_PREFIX = (
    BTV_5PRIME_UTR + BTV_EGFP_CDS + BTV_3PRIME_UTR_PREFIX
)
BTV_REPORTER_CONTEXT_VERSION = "btv-reporter-v1"


class RNAStabilitySiegel(BenchmarkDataset):
    """3' UTR fragment stability MPRA from Siegel et al."""

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize a Siegel stability MPRA dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from the source CSV.
        """
        if type(self) is RNAStabilitySiegel:
            raise TypeError("RNAStabilitySiegel is an abstract class.")

        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=HF_ROOT + self.METADATA.dataset_name + ".parquet",
        )
        embedding_dir_name = (
            f"embeddings-{BTV_REPORTER_CONTEXT_VERSION}"
        )
        self.embedding_dir = str(
            Path(self.dataset_path, embedding_dir_name)
        )
        Path(self.embedding_dir).mkdir(exist_ok=True)
        self.data_df, changed = self._with_reporter_context(self.data_df)
        if changed:
            print("Updating the BTV EGFP reporter context...")
            self.save_processed_df(self.data_df)

    @classmethod
    def _with_reporter_context(
        cls,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, bool]:
        """Add the fixed BTV reporter transcript around each test fragment."""
        dataframe = dataframe.copy()
        sequences = (
            dataframe["sequence"]
            .astype(str)
            .str.upper()
            .str.replace("U", "T", regex=False)
        )
        has_prefix = sequences.str.startswith(BTV_REPORTER_PREFIX)
        has_suffix = sequences.str.endswith(BTV_3PRIME_UTR_SUFFIX)
        has_context = has_prefix & has_suffix

        if has_context.any() and not has_context.all():
            raise ValueError(
                "Siegel dataset mixes fragment-only and reporter-context "
                "sequences."
            )

        if has_context.all():
            changed = "utr_fragment" in dataframe
            if changed:
                dataframe.drop(columns="utr_fragment", inplace=True)
            return dataframe, changed

        dataframe["sequence"] = sequences.map(
            lambda sequence: "".join((
                BTV_REPORTER_PREFIX,
                sequence,
                BTV_3PRIME_UTR_SUFFIX,
            ))
        )

        cds_prefix = np.zeros(len(BTV_5PRIME_UTR), dtype=np.int8)
        cds_coding = np.tile(
            np.array([1, 0, 0], dtype=np.int8),
            len(BTV_EGFP_CDS) // 3,
        )
        fragment_lengths = sequences.str.len()
        track_cache = {}
        for fragment_length in fragment_lengths.unique():
            utr_length = sum((
                len(BTV_3PRIME_UTR_PREFIX),
                int(fragment_length),
                len(BTV_3PRIME_UTR_SUFFIX),
            ))
            cds = np.concatenate([
                cds_prefix,
                cds_coding,
                np.zeros(utr_length, dtype=np.int8),
            ])
            track_cache[int(fragment_length)] = (
                cds,
                np.zeros(len(cds), dtype=np.int8),
            )

        dataframe["cds"] = fragment_lengths.map(
            {
                length: tracks[0]
                for length, tracks in track_cache.items()
            }
        )
        dataframe["splice"] = fragment_lengths.map(
            {
                length: tracks[1]
                for length, tracks in track_cache.items()
            }
        )
        return dataframe, True

    def get_vep_pairs(
        self,
        dataframe: pd.DataFrame,
        value_columns: tuple[str, ...] = ("sequence", "cds", "splice"),
    ) -> pd.DataFrame:
        """Pair measured alternate fragments with their references."""
        references = dataframe[
            dataframe["description"].eq("wild-type")
        ].set_index("oligo_id")
        has_effect = dataframe["target_effect"].notna()
        variants = dataframe[
            ~dataframe["description"].eq("wild-type") & has_effect
        ].copy()
        parent_ids = set(variants["parent_oligo_id"])
        missing = sorted(parent_ids - set(references.index))
        if missing:
            raise ValueError(
                "Missing reference oligos: {}".format(
                    ", ".join(missing[:5])
                )
            )
        for column in value_columns:
            variants[f"ref_{column}"] = variants["parent_oligo_id"].map(
                references[column]
            )
            variants[f"alt_{column}"] = variants[column]
        return variants.reset_index(drop=True)

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Download and process the sequence-level fast-UTR measurements."""
        cell_line = self.METADATA.dataset_name.rsplit("-", 1)[-1]
        filename, checksum = SOURCE_FILES[cell_line]
        source = pd.read_csv(
            download_file(
                SOURCE_ROOT + filename,
                self.raw_data_dir,
                force_redownload=self.force_rebuild_raw,
                expected_sha256=checksum,
            ),
            low_memory=False,
        )

        sequence = (
            source["seq"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("U", "T", regex=False)
        )
        coordinates = source["region"].str.extract(
            r"^(?P<gene>[^|]+)\|"
            r"(?P<chromosome>[^:]+):"
            r"\d+-\d+$"
        )
        is_reference = source["iscontrol"].eq(1)
        is_natural_variant = ~is_reference & source["issnp"].eq(1)
        target = source["ratios_T4T0"].astype(np.float32)
        target_effect = source["effect_size_T4T0"].astype(np.float32).where(
            ~is_reference
        )

        sequence_lengths = sequence.str.len()
        zero_tracks = {
            length: np.zeros(length, dtype=np.int8)
            for length in sequence_lengths.unique()
        }
        dataframe = pd.DataFrame({
            "oligo_id": source["ids"].astype(str),
            "gene": coordinates["gene"].astype(str),
            "chromosome": coordinates["chromosome"].astype(str),
            "sequence": sequence,
            "cds": sequence_lengths.map(zero_tracks),
            "splice": sequence_lengths.map(zero_tracks),
            "target": target,
            "target_reference": target.where(is_reference),
            "target_effect": target_effect,
            "parent_oligo_id": (
                source["parent_control_oligo"].fillna("").astype(str)
            ),
            "description": np.select(
                [is_reference, is_natural_variant],
                ["wild-type", "natural-variant"],
                default="designed-mutant",
            ),
        })
        effect_parents = set(
            dataframe.loc[
                dataframe["target_effect"].notna(),
                "parent_oligo_id",
            ]
        )
        used = dataframe[
            ["target", "target_effect"]
        ].notna().any(axis=1)
        used |= dataframe["oligo_id"].isin(effect_parents)
        dataframe = dataframe[used].reset_index(drop=True)
        dataframe, _ = self._with_reporter_context(dataframe)
        return dataframe


class RNAStabilitySiegelJurkat(RNAStabilitySiegel):
    """Siegel 3' UTR fragment stability measurements in Jurkat cells."""

    METADATA = DatasetMetadata(
        dataset_name="rna-stability-siegel-jurkat",
        species="human",
        task=["regression"],
        target_col=[
            "target",
            "target_reference",
        ],
        default_split_type="chromosome",
        benchmark_set="extended",
        evaluations=("linear_probe", "embedding_vep", "likelihood_vep"),
        variant_region="utr",
        vep_target_col="target_effect",
    )


class RNAStabilitySiegelBeas2B(RNAStabilitySiegel):
    """Siegel 3' UTR fragment stability measurements in BEAS-2B cells."""

    METADATA = DatasetMetadata(
        dataset_name="rna-stability-siegel-beas2b",
        species="human",
        task=["regression"],
        target_col=[
            "target",
            "target_reference",
        ],
        default_split_type="chromosome",
        benchmark_set="extended",
        evaluations=("linear_probe", "embedding_vep", "likelihood_vep"),
        variant_region="utr",
        vep_target_col="target_effect",
    )
