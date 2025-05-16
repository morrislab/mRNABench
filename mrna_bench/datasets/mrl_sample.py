from typing import Any, cast

import gzip
import os
import pathlib
import shutil
import tarfile

import numpy as np
import pandas as pd

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.utils import download_file


EXP_ACCESSIONS = {
    "egfp_unmod_1": "GSM3130435",
    "egfp_unmod_2": "GSM3130436",
    "egfp_pseudo_1": "GSM3130437",
    "egfp_pseudo_2": "GSM3130438",
    "egfp_m1pseudo_1": "GSM3130439",
    "egfp_m1pseudo_2": "GSM3130440",
    "mcherry_unmod_1": "GSM3130441",
    "mcherry_unmod_2": "GSM3130442",
    "designed_library": "GSM3130443",
    "varying_length_25to100": "GSM4084997"
}

M_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114002&format=file"

PRIMER_SEQ = "GGGACATCGTAGAGAGTCGTACTTA"

START_CODON = "ATG"

EGFP_CDS = (
    "atgggcgaattaagtaagggcgaggagctgttcaccgg"
    "ggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcg"
    "agggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctg"
    "cccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccc"
    "cgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgca"
    "ccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacacc"
    "ctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaa"
    "gctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatca"
    "aggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccag"
    "cagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtc"
    "caagctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccg"
    "ccgggatcactctcggcatggacgagctgtacaagttcgaataaagctagcgcctcgactgtgcc"
    "ttctagttgccagccatctgttgtttg"
).upper()

MCHERRY_CDS = (
    "atgcctcccgagaagaagatcaagagcgtgagcaaggg"
    "cgaggaggataacatggccatcatcaaggagttcatgcgcttcaaggtgcacatggagggctccg"
    "tgaacggccacgagttcgagatcgagggcgagggcgagggccgcccctacgagggcacccagacc"
    "gccaagctgaaggtgaccaagggtggccccctgcccttcgcctgggacatcctgtcccctcagtt"
    "catgtacggctccaaggcctacgtgaagcaccccgccgacatccccgactacttgaagctgtcct"
    "tccccgagggcttcaagtgggagcgcgtgatgaacttcgaggacggcggcgtggtgaccgtgacc"
    "caggactcctccctgcaggacggcgagttcatctacaaggtgaagctgcgcggcaccaacttccc"
    "ctccgacggccccgtaatgcagaagaagaccatgggctgggaggcctcctccgagcggatgtacc"
    "ccgaggacggcgccctgaagggcgagatcaagcagaggctgaagctgaaggacggcggccactac"
    "gacgctgaggtcaagaccacctacaaggccaagaagcccgtgcagctgcccggcgcctacaacgt"
    "caacatcaagttggacatcacctcccacaacgaggactacaccatcgtggaacagtacgaacgcg"
    "ccgagggccgccactccaccggcggcatggacgagctgtacaagtcttaacgcctcgactgtgcc"
    "ttctagttgccagccatctgttgtttg"
).upper()

KOZAK_RULES = {
    "strong": [['A', 'G'], ['C', 'A'], ['C', 'G', 'A']],
    "weak": [['T'], ['G'], ['T', 'C']],
}


class MRLSample(BenchmarkDataset):
    """Mean Ribosome Load Dataset from Sample et al. 2019.

    Dataset contains an MPRA for randomized and designed 5'UTRs on human cells.
    Measured output is the mean ribosome load for each sequence.

    The first set of experiments contain random 50-mer 5'UTRs that are inserted
    before an eGFP reporter gene. These experiments are repeated with different
    RNA chemistries (pseudouridine, 1-methylpseudouridine). The second set of
    experiments contain random 50-mer 5'UTRs that are inserted before an
    mCherry reporter gene. Each of these experiments are repeated twice, and
    the mean value of the mean ribosome load is used.

    Finally, a set of designed 5'UTRs with natural occuring SNVs are inserted
    before an eGFP reporter.

    This class is a superclass which is inherited by the specific experiments.
    """

    def __init__(
        self,
        dataset_name: str,
        force_redownload: bool = False,
        hf_url: str | None = None
    ):
        """Initialize MRLSample dataset.

        Args:
            dataset_name: Dataset name formatted mrl-sample-{experiment_name}
                where experiment_name is in: {
                    "egfp",
                    "mcherry",
                    "designed",
                    "varying"
                }.
            force_redownload: Force raw data download even if pre-existing.
            hf_url: URL to download the dataset from Hugging Face.
        """
        if type(self) is MRLSample:
            raise TypeError("MRLSample is an abstract class.")

        self.exp_target = dataset_name.split("-")[-1]
        assert self.exp_target in ["egfp", "mcherry", "designed", "varying"]

        super().__init__(dataset_name, "synthetic", force_redownload, hf_url)

    def _download_raw_data(self):
        """Download raw data from source."""
        self.raw_data_files = []

        dlflag = self.raw_data_dir + "/downloaded"

        if os.path.exists(dlflag) and not self.force_redownload:
            files = pathlib.Path(self.raw_data_dir).glob("*.csv")
            self.raw_data_files = [str(file.absolute()) for file in files]
            return

        print("Downloading data...")

        archive_path = download_file(M_URL, self.raw_data_dir)
        archive_name = self.raw_data_dir + "/GSE114002_RAW.tar"
        os.rename(archive_path, self.raw_data_dir + "/GSE114002_RAW.tar")

        tarfile.open(archive_name).extractall(self.raw_data_dir)
        os.remove(archive_name)

        # Removes unrelated files
        for file in pathlib.Path(self.raw_data_dir).glob("*.csv.gz"):
            if self.exp_target not in file.name:
                os.remove(file)

        print("Extracting data...")
        for file in pathlib.Path(self.raw_data_dir).glob("*.gz"):
            data_path = str(file.absolute())
            with gzip.open(data_path, "rb") as f_in:
                with open(data_path.replace(".gz", ""), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(data_path)

            self.raw_data_files.append(data_path.replace(".gz", ""))

        open(dlflag, "w").close()

    def _process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        print("Processing data...")
        main_df = pd.DataFrame()

        for file in self.raw_data_files:
            if self.exp_target not in file.split("/")[-1]:
                continue

            df = pd.read_csv(file)

            # Remove extra nucleotides from UTR sequences
            if self.exp_target != "varying":
                df["utr"] = df["utr"].str[:50]

            df.set_index("utr", inplace=True)

            exp_file = file.split("/")[-1]
            exp_name = "_".join(exp_file.replace(".csv", "").split("_")[1:])

            df = df[["rl"]].rename(columns={"rl": exp_name})

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how="inner")

        # Takes mean between replicates
        out_df = main_df.T.groupby(
            main_df.columns.str.split('_').str[:-1].str.join('_')
        ).mean().T

        out_df.reset_index(inplace=True)

        # Add flanking sequences
        if self.exp_target == "mcherry":
            out_df["sequence"] = PRIMER_SEQ + out_df["utr"] + MCHERRY_CDS
        else:
            out_df["sequence"] = PRIMER_SEQ + out_df["utr"] + EGFP_CDS

        out_df["cds"] = out_df["utr"].apply(cast(Any, self._get_cds_track))
        out_df["splice"] = out_df["cds"].apply(lambda x: np.zeros_like(x))

        out_df.drop(columns=["utr"], inplace=True)

        d_cols = ["sequence", "cds", "splice"]

        t_prefix = "target_mrl_"
        cols = [t_prefix + c if c not in d_cols else c for c in out_df.columns]
        out_df.columns = pd.Index(cols)

        # Shuffle columns
        cols = d_cols + [c for c in out_df.columns if c not in d_cols]
        out_df = out_df[cols]

        return out_df

    def _get_cds_track(self, utr: str) -> np.ndarray:
        """Get CDS track for all sequences.

        Hard-coded numbers obtained by taking longest ORF in sequence.

        Args:
            utr: UTR sequence.

        Returns:
            Binary track encoding start position of each codon in CDS.
        """
        if self.exp_target == "egfp":
            n_codons = int(732 / 3)
            len_downstream = len(EGFP_CDS) - n_codons * 3
        else:
            n_codons = int(738 / 3)
            len_downstream = len(MCHERRY_CDS) - n_codons * 3

        cds = np.array([1, 0, 0] * n_codons, dtype=np.int8)
        downstream = np.zeros((len_downstream), dtype=np.int8)

        upstream = np.zeros((len(PRIMER_SEQ) + len(utr)), dtype=np.int8)

        cds_track = np.concatenate([upstream, cds, downstream])

        return cds_track

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        self._download_raw_data()
        return self._process_raw_data()


class MRLSampleEGFP(MRLSample):
    """Concrete class for MRL Sample for egfp experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleEGFP dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "mrl-sample-egfp",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mrl-sample/resolve/main/mrl-sample-egfp.parquet"
            )
        )

    def _process_raw_data(self) -> pd.DataFrame:
        """Add post-processing for extra feature columns.

        Add post-processing to add following columns to the dataframe:
            - u_start: upstream start codon presence
            - u_oof_start: out-of-frame upstream start codon presence
            - kozak_quality: One of "strong", "weak", "mixed"
        """
        # Call super class processing first
        df = super()._process_raw_data()

        # Extract utr sequence
        df['utr'] = df.apply(
            lambda x: x['sequence'][len(PRIMER_SEQ):-len(EGFP_CDS)],
            axis=1
        )

        # Classify each utr as having an upstream start codon, and whether
        # that start codon is out-of-frame
        df[['u_start', 'u_oof_start']] = df['utr'].apply(
            lambda x: pd.Series(self._has_upstream_start(x))
        )

        # Classify the strength of the Kozak prefix in each utr
        df['kozak_quality'] = df['utr'].apply(self._kozak_quality)

        df.drop(columns=["utr"], inplace=True)

        return df

    @staticmethod
    def _has_upstream_start(utr: str) -> tuple:
        """Look for upstream start codons, and whether they are out-of-frame.

        Args:
            utr: utr sequence.

        Returns:
            Tuple of (bool, bool), indicating whether an upstream start codon
            exists, and whether it is also out-of-frame
        """
        # Find all start codon positions
        atg_positions = []
        for i in range(len(utr) - 2):
            if utr[i:i + 3].upper() == START_CODON:
                atg_positions.append(i)

        has_upstream_start = len(atg_positions) > 0
        has_oof_start = False
        # Check if any ATG is out of frame relative to CDS
        for pos in atg_positions:
            # Distance from end of UTR to ATG start
            dist_to_end = len(utr) - pos
            # If not divisible by 3, it's out of frame
            if dist_to_end % 3 != 0:
                has_oof_start = True
        return int(has_upstream_start), int(has_oof_start)

    @staticmethod
    def _kozak_quality(utr: str) -> str:
        """
        Classifies the Kozak sequence quality from the last 3 bases of the utr.

        Args:
            utr: utr sequence

        Returns:
            "strong", "weak", or "mixed" based on positional base preferences.
        """
        # Since CDS is constant in this data
        # only the last 3-mer of the UTR matters
        kozak_prefix = utr[-3:]

        def matches(rule):
            return all(
                base in matching_bases
                for base, matching_bases in
                zip(kozak_prefix, KOZAK_RULES[rule])
            )

        strong = matches("strong")
        weak = matches("weak")

        # If it falls into both buckets, or neither -
        # this is a "mixed" or "in between" sequence
        if strong == weak:
            return "mixed"
        return "strong" if strong else "weak"


class MRLSampleMCherry(MRLSample):
    """Concrete class for MRL Sample for mCherry experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleMCherry dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "mrl-sample-mcherry",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mrl-sample/resolve/main/mrl-sample-mcherry.parquet"
            )
        )


class MRLSampleDesigned(MRLSample):
    """Concrete class for MRL Sample for designed experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleDesigned dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "mrl-sample-designed",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mrl-sample/resolve/main/mrl-sample-designed.parquet"
            )
        )


class MRLSampleVarying(MRLSample):
    """Concrete class for MRL Sample for varying length experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleVarying dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            "mrl-sample-varying",
            force_redownload,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "mrl-sample/resolve/main/mrl-sample-varying.parquet"
            )
        )
