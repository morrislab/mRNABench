from collections.abc import Callable

from mrna_bench.data_splitter.data_splitter import DataSplitter
from mrna_bench.data_splitter.kmer_split import KMerSplitter
from mrna_bench.data_splitter.homology_split import HomologySplitter
from mrna_bench.data_splitter.sklearn_split import SklearnSplitter
from mrna_bench.data_splitter.chromosome_split import ChromosomeSplitter

SPLIT_CATALOG: dict[str, Callable[..., DataSplitter]] = {
    "default": SklearnSplitter,
    "homology": HomologySplitter,
    "kmer": KMerSplitter,
    "chromosome": ChromosomeSplitter
}
