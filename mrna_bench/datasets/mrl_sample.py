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

    def __init__(self, dataset_name: str, force_redownload: bool = False):
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
        """
        if type(self) is MRLSample:
            raise TypeError("MRLSample is an abstract class.")

        self.exp_target = dataset_name.split("-")[-1]
        assert self.exp_target in ["egfp", "mcherry", "designed", "varying"]

        super().__init__(dataset_name, ["human"], force_redownload)

    def get_raw_data(self):
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

    def process_raw_data(self) -> pd.DataFrame:
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

        out_df["cds"] = out_df["utr"].apply(cast(Any, self.get_cds_track))
        out_df["splice"] = out_df["cds"].apply(lambda x: np.zeros_like(x))

        out_df.drop(columns=["utr"], inplace=True)

        d_cols = ["sequence", "cds", "splice"]
        cols = ["mrl_" + c if c not in d_cols else c for c in out_df.columns]
        out_df.columns = pd.Index(cols)

        return out_df

    def get_cds_track(self, utr: str) -> np.ndarray:
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


class MRLSampleEGFP(MRLSample):
    """Concrete class for MRL Sample for egfp experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleEGFP dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__("mrl-sample-egfp", force_redownload)


class MRLSampleMCherry(MRLSample):
    """Concrete class for MRL Sample for mCherry experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleMCherry dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__("mrl-sample-mcherry", force_redownload)


class MRLSampleDesigned(MRLSample):
    """Concrete class for MRL Sample for designed experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleDesigned dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__("mrl-sample-designed", force_redownload)


class MRLSampleVarying(MRLSample):
    """Concrete class for MRL Sample for varying length experiments."""

    def __init__(self, force_redownload=False):
        """Initialize MRLSampleVarying dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__("mrl-sample-varying", force_redownload)
