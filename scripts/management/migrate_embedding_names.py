"""Add embedding configs to embeddings saved before config hashing.

Renames::

    {data_dir}/{dataset}/embeddings/{dataset}_{model}[_{i}-{n}].{npz|h5}

to ``{dataset}_{model}_{config_hash}[_{i}-{n}].{npz|h5}`` and stores the
config in the file. Old files used mean pooling of the final layer. Naive
baselines named ``{model}-k{kmers}`` counted only
those k-mer lengths. The attention implementation is the model's current
default.

Usage
-----
python scripts/management/migrate_embedding_names.py --data_dir DIR --dry-run
python scripts/management/migrate_embedding_names.py --data_dir DIR \\
    --fallback_model Orthrus
"""

import argparse
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

from mrna_bench.embedder import get_config_hash, get_embedding_config
from mrna_bench.models import MODEL_CATALOG
from mrna_bench.models.pooling import MeanPool

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument(
    "--fallback_model",
    type=str,
    default=None,
    help="Catalog model whose defaults apply to names not in the catalog.",
)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

KMER_SUFFIX = re.compile(r"^(naive-\d)-k(\d+)$")


def write_config(path: Path, config: dict) -> None:
    """Store config in an embedding file without rewriting its arrays."""
    config_str = json.dumps(config)
    if path.suffix == ".h5":
        with h5py.File(path, "a") as f:
            f.attrs["config"] = config_str
        return
    buf = BytesIO()
    np.save(buf, np.array(config_str))
    with zipfile.ZipFile(path, "a") as f:
        f.writestr("config.npy", buf.getvalue())


if __name__ == "__main__":
    short_names = {
        cls.get_model_short_name(version): cls
        for cls in MODEL_CATALOG.values()
        for version in cls.valid_versions
    }
    fallback = MODEL_CATALOG[args.fallback_model] if args.fallback_model else None

    skipped = []
    for path in sorted(Path(args.data_dir).glob("*/embeddings/*")):
        parts = path.stem.split("_")
        if path.suffix not in (".npz", ".h5") or len(parts) not in (2, 3):
            continue
        if len(parts) == 3 and not re.fullmatch(r"\d+-\d+", parts[2]):
            continue  # already migrated

        dataset_name, model_name = parts[:2]
        pooling = MeanPool()

        kmer_match = KMER_SUFFIX.match(model_name)
        if kmer_match:
            model_name = kmer_match.group(1)

        model_class = short_names.get(model_name, fallback)
        if model_class is None:
            skipped.append(path)
            continue

        config = get_embedding_config(model_class, pooling)
        if kmer_match:
            kmer_list = [int(k) for k in kmer_match.group(2)]
            config["model_options"] = {"kmer_list": kmer_list}
        new_parts = [dataset_name, model_name, get_config_hash(config)]
        new_path = path.with_name("_".join(new_parts + parts[2:]) + path.suffix)

        print("{} -> {}".format(path.name, new_path.name))
        if not args.dry_run:
            write_config(path, config)
            path.rename(new_path)

    for path in skipped:
        print("skipped, model not in catalog: {}".format(path))
