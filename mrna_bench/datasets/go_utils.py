import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mrna_bench.datasets.dataset_utils import (
    create_cds_track,
    create_sequence,
    create_splice_track,
)
from mrna_bench.utils import download_file


SOURCE_FILE = "go_annotations.tsv.gz"
SOURCE_URL = (
    "https://github.com/morrislab/mRNABench/raw/main/"
    f"resources/{SOURCE_FILE}"
)
SOURCE_SHA256 = (
    "0cef53a886e4dfd088e806dc504d494a7c6238e9225012bc0ca211ee37e7049d"
)
GO_TERMS = {
    "mf": [
        "GO:0004672", "GO:0008201", "GO:0003723", "GO:0005102",
        "GO:0019904", "GO:0019899", "GO:0046983", "GO:0051015",
        "GO:0000981", "GO:0005516", "GO:0042802", "GO:0020037",
        "GO:0003677", "GO:0042393", "GO:0003714", "GO:0005178",
        "GO:0004888", "GO:0005543", "GO:0044325", "GO:0003713",
    ],
    "bp": [
        "GO:0007267", "GO:0009410", "GO:0001666", "GO:0007186",
        "GO:0016477", "GO:0035556", "GO:0007204", "GO:0001525",
        "GO:0098609", "GO:0000226", "GO:0007166", "GO:0006897",
        "GO:0030154", "GO:0007283", "GO:0045087", "GO:0030036",
        "GO:0006914", "GO:0008360", "GO:0000278", "GO:0002250",
    ],
    "cc": [
        "GO:0030426", "GO:0031012", "GO:0043025", "GO:0016607",
        "GO:0005635", "GO:0000151", "GO:0000786", "GO:0043197",
        "GO:0000781", "GO:0005681", "GO:0032587", "GO:0016605",
        "GO:0036064", "GO:0000776", "GO:0005938", "GO:0045202",
        "GO:0005813", "GO:0009897", "GO:0005886", "GO:0090575",
    ],
}


def _source_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return checksum == SOURCE_SHA256


def build_go_dataset(branch: str, raw_data_dir: str) -> pd.DataFrame:
    """Build one GO branch from frozen annotations and GENCODE v41."""
    if branch not in GO_TERMS:
        raise ValueError(f"Unsupported GO branch: {branch}")

    repository_root = Path(__file__).resolve().parents[2]
    source_path = repository_root / "resources" / SOURCE_FILE
    if not _source_is_valid(source_path):
        source_path = Path(
            download_file(
                SOURCE_URL,
                raw_data_dir,
                expected_sha256=SOURCE_SHA256,
            )
        )

    annotations = pd.read_csv(
        source_path,
        sep="\t",
        keep_default_na=False,
    )
    annotations = annotations.loc[annotations["branch"] == branch]

    try:
        import genome_kit as gk
    except ImportError:
        print(
            "GenomeKit is required for raw processing with GENCODE v41. "
            "Install the mrna-bench dev dependencies."
        )
        raise

    genome = gk.Genome("gencode.v41")
    transcript_ids = set(annotations["transcript_id"])
    transcripts = {
        transcript.id.split(".")[0]: transcript
        for transcript in genome.transcripts
        if transcript.id.split(".")[0] in transcript_ids
    }
    missing = sorted(transcript_ids - set(transcripts))
    if missing:
        raise ValueError(
            f"{len(missing)} transcripts are absent from GENCODE v41: "
            f"{missing[:5]}"
        )

    term_index = {
        term: index for index, term in enumerate(GO_TERMS[branch])
    }
    rows = []
    for row in annotations.itertuples(index=False):
        transcript = transcripts[row.transcript_id]
        target = np.zeros(len(term_index), dtype=np.int8)
        for term in row.go_ids.split("|") if row.go_ids else ():
            target[term_index[term]] = 1
        rows.append({
            "gene": row.gene,
            "chromosome": transcript.chrom.removeprefix("chr"),
            "sequence": create_sequence(transcript, genome).upper(),
            "cds": create_cds_track(transcript),
            "splice": create_splice_track(transcript),
            "target": target,
        })
    return pd.DataFrame(rows)
