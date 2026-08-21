import pandas as pd

from mrna_bench.datasets.benchmark_dataset import (
    BenchmarkDataset,
    DatasetMetadata,
)
from mrna_bench.utils import download_file


SOURCE_COMMIT = "d987ee5c06f88d57870e10bd63dd6bfa397af866"
SOURCE_URL = (
    "https://raw.githubusercontent.com/deepgenomics/"
    f"UTR_variants_DL_manuscript/{SOURCE_COMMIT}/data/"
)


class UTRVariantsBohn(BenchmarkDataset):
    """UTR Variants dataset from Bohn et al."""

    def get_vep_pairs(
        self,
        dataframe: pd.DataFrame,
        value_columns: tuple[str, ...] = ("sequence", "cds", "splice"),
    ) -> pd.DataFrame:
        """Pair each variant with its transcript's wild-type row.

        Args:
            dataframe: Bohn rows containing wild-type and variant sequences.
            value_columns: Columns to copy into ref- and alt-prefixed output
                columns.

        Returns:
            Variant rows with aligned reference and alternate values.
        """
        is_reference = dataframe["description"].eq("wild-type")
        references = dataframe[is_reference].set_index("transcript_id")
        variants = dataframe[~is_reference].copy()
        missing = sorted(
            set(variants["transcript_id"]) - set(references.index)
        )
        if missing:
            raise ValueError(
                "Missing wild-type for transcripts: {}".format(
                    ", ".join(map(str, missing))
                )
            )
        for column in value_columns:
            variants[f"ref_{column}"] = variants["transcript_id"].map(
                references[column]
            )
            variants[f"alt_{column}"] = variants[column]
        return variants.reset_index(drop=True)

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
        hf_url: str = "",
    ):
        """Initialize UTRVariantsBohn dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
            hf_url: URL to download the dataset from Hugging Face.
        """
        if type(self) is UTRVariantsBohn:
            raise TypeError("UTRVariantsBohn is an abstract class.")

        self.utr_type = self.METADATA.dataset_name.split("-")[-1]
        assert self.utr_type in ["utr5", "utr3"]

        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=hf_url,
        )

    def _get_data_from_raw(self) -> pd.DataFrame:
        """Download and process the Bohn UTR variant benchmark."""
        import numpy as np
        try:
            from genome_kit import Genome, VariantGenome
        except ImportError:
            print(
                "GenomeKit is required for raw processing with "
                "NCBI RefSeq v109. Install the mrna-bench dev dependencies."
            )
            raise

        raw_path = download_file(
            f"{SOURCE_URL}{self.utr_type}_plp_benchmark.tsv",
            self.raw_data_dir,
        )
        variants = pd.read_csv(raw_path, sep="\t")
        genome = Genome("ncbi_refseq.v109")
        transcripts = {
            (transcript.id, transcript.chrom): transcript
            for transcript in genome.transcripts
        }

        def interval_lengths(intervals, position=None, delta=0):
            if position is None:
                return [len(interval) for interval in intervals]
            return [
                len(interval) + (
                    delta
                    if interval.start <= position < interval.end
                    else 0
                )
                for interval in intervals
            ]

        def encode(
            source_genome,
            transcript,
            utr3_lengths,
            cds_lengths,
            utr5_lengths,
        ):
            raw_sequence = "".join(
                source_genome.dna(exon) for exon in transcript.exons
            )
            sequence = "".join(
                base if base in "ACGT" else "N"
                for base in raw_sequence
            ).rstrip("N")

            utr5_length = sum(utr5_lengths)
            cds_length = sum(cds_lengths)
            # Preserve the source notebook's published feature-track order.
            regions = utr3_lengths + cds_lengths + utr5_lengths
            total_length = sum(regions)

            cds = np.zeros(total_length, dtype=int)
            cds[
                utr5_length : utr5_length + cds_length : 3
            ] = 1

            splice = np.zeros(total_length, dtype=int)
            cumulative_length = 0
            for length in regions:
                cumulative_length += length
                splice[cumulative_length - 1] = 1
            return sequence, cds, splice

        rows = []
        emitted_references = set()
        for row in variants.to_dict("records"):
            chrom, source_position, ref, alt = row["variant"].split(":")
            position = int(source_position) - 1
            transcript = transcripts[(row["transcript_id"], chrom)]
            common = {
                "transcript_id": transcript.id,
                "gene": transcript.gene.name,
                "chromosome": chrom.strip("chr"),
            }

            if transcript.id not in emitted_references:
                sequence, cds, splice = encode(
                    genome,
                    transcript,
                    interval_lengths(transcript.utr3s),
                    interval_lengths(transcript.cdss),
                    interval_lengths(transcript.utr5s),
                )
                rows.append({
                    **common,
                    "sequence": sequence,
                    "cds": cds,
                    "splice": splice,
                    "target": 0,
                    "description": "wild-type",
                })
                emitted_references.add(transcript.id)

            variant_genome = VariantGenome(
                genome,
                genome.variant(
                    f"{chrom}:{position + 1}:{ref}:{alt}"
                ),
            )
            delta = len(alt) - len(ref)
            utr3_lengths = interval_lengths(
                transcript.utr3s,
                position,
                delta,
            )
            cds_lengths = interval_lengths(transcript.cdss)
            utr5_lengths = interval_lengths(transcript.utr5s)
            if self.utr_type == "utr5":
                cds_lengths = interval_lengths(
                    transcript.cdss,
                    position,
                    delta,
                )
                utr5_lengths = interval_lengths(
                    transcript.utr5s,
                    position,
                    delta,
                )

            sequence, cds, splice = encode(
                variant_genome,
                transcript,
                utr3_lengths,
                cds_lengths,
                utr5_lengths,
            )
            description = f"{chrom}:{position} {ref}:{alt}"
            if pd.notna(row["proposed_mechanism"]):
                description += f",{row['proposed_mechanism']}"
            rows.append({
                **common,
                "sequence": sequence,
                "cds": cds,
                "splice": splice,
                "target": row["class"],
                "description": description,
            })

        return pd.DataFrame(rows)


class UTRVariantsBohnUTR5(UTRVariantsBohn):
    """Concrete class for UTR Variants dataset (5' UTR)."""

    METADATA = DatasetMetadata(
        dataset_name="utr-variants-bohn-utr5",
        species="human",
        task=["classification"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe", "embedding_vep", "likelihood_vep"),
        variant_region="utr",
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize UTRVariantsBohnUTR5 dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "utr-variants-bohn/resolve/main/"
                "utr-variants-bohn-utr5.parquet"
            ),
        )


class UTRVariantsBohnUTR3(UTRVariantsBohn):
    """Concrete class for UTR Variants dataset (3' UTR)."""

    METADATA = DatasetMetadata(
        dataset_name="utr-variants-bohn-utr3",
        species="human",
        task=["classification"],
        target_col=["target"],
        default_split_type="default",
        benchmark_set="core",
        evaluations=("linear_probe", "embedding_vep", "likelihood_vep"),
        variant_region="utr",
    )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
    ):
        """Initialize UTRVariantsBohnUTR3 dataset.

        Args:
            force_redownload_hf: Force redownload from HuggingFace.
            force_rebuild_raw: Force rebuild from raw data source.
        """
        super().__init__(
            force_redownload_hf=force_redownload_hf,
            force_rebuild_raw=force_rebuild_raw,
            hf_url=(
                "https://huggingface.co/datasets/morrislab/"
                "utr-variants-bohn/resolve/main/"
                "utr-variants-bohn-utr3.parquet"
            ),
        )
