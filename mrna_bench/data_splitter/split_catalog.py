from collections.abc import Callable

from mrna_bench.data_splitter.data_splitter import DataSplitter
from mrna_bench.data_splitter.ss_split import SSSplitter
from mrna_bench.data_splitter.homology_split import HomologySplitter
from mrna_bench.data_splitter.sklearn_split import SklearnSplitter


SPLIT_CATALOG: dict[str, Callable[..., DataSplitter]] = {
    "default": SklearnSplitter,
    "homology": HomologySplitter,
    "ss": SSSplitter,
}