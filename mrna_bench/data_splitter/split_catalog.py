from mrna_bench.data_splitter.homology_split import HomologySplitter
from mrna_bench.data_splitter.sklearn_split import SklearnSplitter


SPLIT_CATALOG = {
    "default": SklearnSplitter,
    "homology": HomologySplitter
}
