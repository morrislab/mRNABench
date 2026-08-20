from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

import pandas as pd

from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG
from mrna_bench.utils import download_file, get_data_path


VALID_TASKS = (
    "regression",
    "classification",
    "multilabel",
)

if TYPE_CHECKING:
    from mrna_bench.models import EmbeddingModel


class EvaluationMethod(StrEnum):
    """Evaluation routes supported by a dataset."""

    LINEAR_PROBE = "linear_probe"
    EMBEDDING_VEP = "embedding_vep"
    LIKELIHOOD_VEP = "likelihood_vep"


@dataclass(frozen=True)
class TaskSpec:
    """One supported target and its biological prediction task."""

    task: str
    target_col: str


@dataclass(frozen=True)
class DatasetMetadata:
    """Required metadata for all benchmark datasets."""

    dataset_name: str
    species: str
    task: list[str]
    target_col: list[str]
    default_split_type: str
    benchmark_set: str
    evaluations: tuple[EvaluationMethod | str, ...]
    variant_region: str | None = None

    @property
    def is_vep(self) -> bool:
        """Return whether the dataset defines a VEP evaluation route.

        Returns:
            True if embedding- or likelihood-based VEP is supported.
        """
        vep_methods = {
            EvaluationMethod.EMBEDDING_VEP,
            EvaluationMethod.LIKELIHOOD_VEP,
        }
        return bool(vep_methods.intersection(self.evaluations))

    def __post_init__(self):
        """Post-initialization checks."""
        enforced_fields = {
            "benchmark_set": ("core", "extended"),
            "default_split_type": tuple(SPLIT_CATALOG.keys()),
        }

        for field_name, options in enforced_fields.items():
            value = getattr(self, field_name)
            if value not in options:
                raise ValueError(
                    "{} must be one of {}".format(field_name, options)
                )

        for t in self.task:
            if t not in VALID_TASKS:
                raise ValueError(
                    "task elements must be one of {}".format(VALID_TASKS)
                )

        if not self.task or not self.target_col:
            raise ValueError("task and target_col must not be empty.")
        valid_cardinality = any((
            len(self.task) == len(self.target_col),
            len(self.task) == 1,
            len(self.target_col) == 1,
        ))
        if not valid_cardinality:
            raise ValueError(
                "task and target_col must have equal lengths, or one side "
                "must contain a single value."
            )
        evaluations = tuple(
            EvaluationMethod(evaluation)
            for evaluation in self.evaluations
        )
        if not evaluations:
            raise ValueError("evaluations must not be empty.")
        object.__setattr__(self, "evaluations", evaluations)
        if self.variant_region not in {None, "full", "utr", "cds"}:
            raise ValueError(
                "variant_region must be one of full, utr, cds, or None."
            )

    @property
    def task_specs(self) -> tuple[TaskSpec, ...]:
        """Return explicit task/target combinations.

        Returns:
            Supported biological task and target-column pairs.
        """
        pairs = (
            zip(self.task, self.target_col)
            if len(self.task) == len(self.target_col)
            else (
                ((task, self.target_col[0]) for task in self.task)
                if len(self.target_col) == 1
                else ((self.task[0], target) for target in self.target_col)
            )
        )
        return tuple(TaskSpec(task, target) for task, target in pairs)

    def compatible_evaluations(
        self,
        model: "EmbeddingModel",
    ) -> tuple[EvaluationMethod, ...]:
        """Return dataset evaluations supported by a model.

        Args:
            model: Loaded embedding model to evaluate for compatibility.

        Returns:
            Dataset evaluation routes supported by the model.
        """
        from mrna_bench.models import ModelBehavior

        compatible: list[EvaluationMethod] = []
        for evaluation in self.evaluations:
            evaluation = EvaluationMethod(evaluation)
            has_embedding = model.supports(ModelBehavior.EMBEDDING)
            has_likelihood = any((
                model.supports(ModelBehavior.CAUSAL_LIKELIHOOD),
                model.supports(ModelBehavior.PSEUDO_LIKELIHOOD),
            ))
            is_linear_probe = evaluation == EvaluationMethod.LINEAR_PROBE
            if is_linear_probe and has_embedding:
                compatible.append(evaluation)
            score_scope = getattr(model, "sequence_score_scope", "full")
            region_compatible = any((
                self.variant_region is None,
                score_scope == "full",
                score_scope == self.variant_region,
            ))
            is_embedding_vep = evaluation == EvaluationMethod.EMBEDDING_VEP
            if all((is_embedding_vep, has_embedding, region_compatible)):
                compatible.append(evaluation)
            is_likelihood_vep = (
                evaluation == EvaluationMethod.LIKELIHOOD_VEP
            )
            if all((is_likelihood_vep, has_likelihood, region_compatible)):
                compatible.append(evaluation)
        return tuple(compatible)


class BenchmarkDataset(ABC):
    """Abstract class for benchmarking datasets.

    Sequences are internally represented as strings. This is less storage
    efficient, but easier to handle as most parts of the pipeline use raw text.
    """

    METADATA: ClassVar[DatasetMetadata]

    def get_vep_pairs(
        self,
        dataframe: pd.DataFrame,
        value_columns: tuple[str, ...] = ("sequence", "cds", "splice"),
    ) -> pd.DataFrame:
        """Return the dataset's variants in canonical ref/alt columns.

        Args:
            dataframe: Dataset rows containing reference and alternate
                variants.
            value_columns: Columns to copy into ref- and alt-prefixed output
                columns.

        Returns:
            One row per variant with aligned reference and alternate values.
        """
        raise NotImplementedError(
            f"{self.dataset_name} does not define VEP row pairing."
        )

    def __init__(
        self,
        force_redownload_hf: bool = False,
        force_rebuild_raw: bool = False,
        hf_url: str = "",
    ):
        """Initialize BenchmarkDataset.

        Args:
            force_redownload_hf: Forces redownload from HuggingFace.
            force_rebuild_raw: Forces rebuild from raw data source.
            hf_url: HuggingFace URL for downloading processed data.
        """
        self.metadata = self.METADATA
        self.dataset_name = self.METADATA.dataset_name

        self.hf_url = hf_url if hf_url else None
        self.force_rebuild_raw = force_rebuild_raw

        self.data_storage_path = get_data_path()
        self.init_folders()

        if force_rebuild_raw:
            print("Rebuilding from raw data source...")
            self.data_df = self._get_data_from_raw()
            self.save_processed_df(self.data_df)
        elif force_redownload_hf:
            if self.hf_url is None:
                raise ValueError(
                    "force_redownload_hf=True but no HuggingFace URL defined."
                )
            print("Downloading data from HuggingFace Hub...")
            self.data_df = self.get_data_hf()
            self.save_processed_df(self.data_df)
        elif not self.load_processed_df():
            if self.hf_url is not None:
                print("Downloading data from HuggingFace Hub...")
                try:
                    self.data_df = self.get_data_hf()
                except Exception as e:
                    print("Error downloading from HuggingFace: {}".format(e))
                    print("Attempting to process from raw data.")
                    self.data_df = self._get_data_from_raw()
            else:
                print("No HF URL provided. Processing from raw.")
                self.data_df = self._get_data_from_raw()
            self.save_processed_df(self.data_df)

    def get_data_hf(self) -> pd.DataFrame:
        """Download processed data from HuggingFace.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        if self.hf_url is None:
            raise ValueError("HuggingFace URL not provided.")

        processed_data_path = download_file(self.hf_url, self.raw_data_dir)
        return pd.read_parquet(processed_data_path)

    @abstractmethod
    def _get_data_from_raw(self) -> pd.DataFrame:
        """Abstract method to download and process data from raw source.

        This class can be implemented in the subclass with a passthrough if
        a huggingface url is instead provided.
        """
        pass

    def subset_df(self, target_cols: list[str]) -> pd.DataFrame:
        """Return invariant sequence columns and requested targets."""
        invariant_cols = {
            "sequence",
            "gene",
            "chromosome",
            "cds",
            "splice",
        }
        keep_cols = invariant_cols.union(target_cols)
        columns = [
            column for column in self.data_df.columns if column in keep_cols
        ]
        return self.data_df[columns]

    def init_folders(self):
        """Initialize folders for storing raw data.

        Creates a structure with:

        - data_path
        |    - dataset_name
        |    |    - raw_data
        |    |    - embeddings
        """
        ds_path = Path(self.data_storage_path) / self.dataset_name
        ds_path.mkdir(exist_ok=True)

        raw_data_dir = Path(ds_path) / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)

        emb_dir = Path(ds_path) / "embeddings"
        emb_dir.mkdir(exist_ok=True)

        self.dataset_path = str(ds_path)
        self.raw_data_dir = str(raw_data_dir)
        self.embedding_dir = str(emb_dir)

    def save_processed_df(self, df: pd.DataFrame):
        """Save dataframe to data storage path.

        Args:
            df: Processed dataframe to save.
        """
        df.to_parquet(self.dataset_path + "/data_df.parquet")

    def load_processed_df(self) -> bool:
        """Load processed dataframe from data storage path.

        Returns:
            Whether dataframe was successfully loaded to class property.
        """
        try:
            df_path = self.dataset_path + "/data_df.parquet"
            self.data_df = pd.read_parquet(df_path)
        except FileNotFoundError:
            print("Processed data frame not found.")
            return False
        return True

    def get_splits(
        self,
        split_ratios: tuple[float, float, float],
        random_seed: int = 2541,
        split_type: str | None = None,
        split_kwargs: dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get data splits for the dataset.

        Args:
            split_ratios: Ratios for train, val, test splits.
            random_seed: Random seed for reproducibility.
            split_type: Type of split to use. Defaults to metadata default.
            split_kwargs: Additional arguments for the split type.

        Returns:
            Dictionary of dataframes containing splits.
        """
        from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

        if split_type is None:
            split_type = self.metadata.default_split_type

        split_kwargs = dict(split_kwargs or {})
        if split_type == "homology" and "species" not in split_kwargs:
            split_kwargs["species"] = self.metadata.species

        splitter = SPLIT_CATALOG[split_type](**split_kwargs)
        splits = splitter.get_all_splits_df(
            self.data_df,
            split_ratios,
            random_seed,
        )

        split_df = {
            "train_df": splits[0],
            "val_df": splits[1],
            "test_df": splits[2],
        }

        return split_df
