import numpy as np

from mrna_bench import load_dataset
from mrna_bench.datasets import BenchmarkDataset
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG
from mrna_bench.embedder import get_embedding_filepath
from mrna_bench.models import EmbeddingModel

from mrna_bench.linear_probe.linear_probe import LinearProbe
from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.linear_probe.evaluator import LinearProbeEvaluator


class LinearProbeBuilder:
    """Factory class for LinearProbeCore."""

    _DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)

    @staticmethod
    def load_persisted_embeddings(
        embedding_dir: str,
        model_short_name: str,
        dataset_name: str,
    ) -> np.ndarray:
        """Load pre-computed embeddings for dataset from persisted location.

        Args:
            embedding_dir: Directory where embedding is stored.
            model_short_name: Shortened name of embedding model version.
            dataset_name: Name of dataset which was embedded.

        Returns:
            Embeddings for dataset computed using embedding model.
        """
        embeddings_fn = get_embedding_filepath(
            embedding_dir,
            model_short_name,
            dataset_name,
        ) + ".npz"

        embeddings = np.load(embeddings_fn)["embedding"]
        return embeddings

    def __init__(
        self,
        dataset: BenchmarkDataset | None = None,
        dataset_name: str | None = None,
    ):
        """Initialize LinearProbeBuilder.

        Can be initialized with BenchmarkDataset instance or name. Only one
        of dataset or dataset_name should be provided.

        Recommended build order:
            1. Fetch embeddings
            2. Set data splitter
            3. Set target column
            4. Build evaluator
            5. (Optional) specify if persister should be used
            6. Build LinearProbe instance.

        Args:
            dataset: BenchmarkDataset to linearly probe.
            dataset_name: Name of dataset to linearly probe.
        """
        if dataset is None and dataset_name is None:
            raise ValueError("Must provide dataset or dataset name.")
        elif dataset is not None and dataset_name is not None:
            raise ValueError("Provide only one of dataset or dataset name.")

        if dataset is None and dataset_name is not None:
            dataset = load_dataset(dataset_name)

        assert dataset is not None

        self.dataset = dataset
        self.data_df = dataset.data_df

        metadata = getattr(dataset, "metadata", None)

        self.target_col = self._resolve_default_target_col(metadata)
        self.task = self._resolve_default_task(metadata)
        self.split_type = self._resolve_default_split_type(metadata)
        self.eval_all_splits = False

        self.model_short_name: str | None = None
        self.embeddings: np.ndarray | None = None
        self.persister_flag = False

        self.splitter = self._build_default_splitter(metadata)
        self.evaluator = LinearProbeEvaluator(self.task)

    @staticmethod
    def _resolve_default_target_col(metadata: object | None) -> str:
        if metadata is not None:
            target_cols = getattr(metadata, "target_col", None)
            if isinstance(target_cols, list) and len(target_cols) > 0:
                return target_cols[0]
        return "target"

    @staticmethod
    def _resolve_default_task(metadata: object | None) -> str:
        if metadata is not None:
            tasks = getattr(metadata, "task", None)
            if isinstance(tasks, list):
                for task in tasks:
                    if task in LinearProbeEvaluator.valid_tasks:
                        return task
        return "regression"

    @staticmethod
    def _resolve_default_split_type(metadata: object | None) -> str:
        if metadata is not None:
            split_type = getattr(metadata, "default_split_type", None)
            if isinstance(split_type, str) and split_type in SPLIT_CATALOG:
                return split_type
        return "default"

    def _build_default_splitter(self, metadata: object | None):
        split_args = {}
        if self.split_type == "homology" and metadata is not None:
            species = getattr(metadata, "species", None)
            if isinstance(species, str) and species != "":
                split_args["species"] = species

        return SPLIT_CATALOG[self.split_type](
            self._DEFAULT_SPLIT_RATIOS,
            **split_args
        )

    def fetch_embedding_by_model_instance(
        self,
        model: EmbeddingModel,
    ) -> "LinearProbeBuilder":
        """Get embeddings for LinearProbe from EmbeddingModel instance.

        Args:
            model: EmbeddingModel instance used to generate embeddings.

        Returns:
            LinearProbeBuilder with set embeddings.
        """
        self.model_short_name = model.short_name

        self.embeddings = self.load_persisted_embeddings(
            self.dataset.embedding_dir,
            self.model_short_name,
            self.dataset.dataset_name,
        )

        return self

    def fetch_embedding_by_model_name(
        self,
        model_short_name: str,
    ) -> "LinearProbeBuilder":
        """Get embeddings for LinearProbe using model short name.

        Args:
            model_short_name: Short name of model used to generate embeddings.

        Returns:
            LinearProbeBuilder with set embeddings.
        """
        self.model_short_name = model_short_name

        self.embeddings = self.load_persisted_embeddings(
            self.dataset.embedding_dir,
            self.model_short_name,
            self.dataset.dataset_name,
        )

        return self

    def fetch_embedding_by_filename(
        self,
        embedding_name: str
    ) -> "LinearProbeBuilder":
        """Get embeddings for LinearProbe using embedding file name.

        Args:
            embedding_name: Name of embedding file.

        Returns:
            LinearProbeBuilder with set embeddings.
        """
        embedding_name = embedding_name.replace(".npz", "")

        emb_fn_arr = embedding_name.split("_")

        if len(emb_fn_arr) < 2:
            raise ValueError(
                "Invalid embedding filename format: {}".format(embedding_name)
            )

        self.model_short_name = emb_fn_arr[1]

        self.embeddings = self.load_persisted_embeddings(
            self.dataset.embedding_dir,
            self.model_short_name,
            self.dataset.dataset_name,
        )

        return self

    def fetch_embedding_by_embedding_instance(
        self,
        model_short_name: str,
        embedding: np.ndarray,
    ) -> "LinearProbeBuilder":
        """Store embeddings for LinearProbe using an embedding instance.

        Args:
            model_short_name: Short name of model used to generate embeddings.
            embedding: Locally generated embedding for dataset.

        Returns:
            LinearProbeBuilder with set embeddings.
        """
        self.model_short_name = model_short_name
        self.embeddings = embedding

        return self

    def build_splitter(
        self,
        split_type: str,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        eval_all_splits: bool = False,
        **split_args
    ) -> "LinearProbeBuilder":
        """Set data splitter for LinearProbe.

        Args:
            split_type: Method used for data split generation.
            split_ratios: Ratio of data split sizes as a fraction of dataset.
            eval_all_splits: Evaluate metrics on all splits. Only evaluates
                validation split otherwise.
            **split_args: Additional arguments for data splitter.

        Returns:
            LinearProbeBuilder with set data splitter.
        """
        self.eval_all_splits = eval_all_splits
        self.split_type = split_type
        self.splitter = SPLIT_CATALOG[split_type](split_ratios, **split_args)

        return self

    def set_target(self, target_col: str) -> "LinearProbeBuilder":
        """Set linear probing target column.

        Args:
            target_col: Column from dataframe to use as labels.

        Returns:
            LinearProbeBuilder with set task and target column.
        """
        self.target_col = target_col

        return self

    def build_evaluator(self, task: str) -> "LinearProbeBuilder":
        """Set evaluator for LinearProbe.

        Args:
            task: Task for linear probing evaluation.

        Returns:
            LinearProbeBuilder with set evaluator.
        """
        self.task = task
        self.evaluator = LinearProbeEvaluator(self.task)

        return self

    def use_persister(self) -> "LinearProbeBuilder":
        """Indicate that persister for LinearProbe should be built.

        Returns:
            LinearProbeBuilder with persister flag set.
        """
        self.persister_flag = True
        return self

    def validate(self) -> list[str]:
        """Return a list of missing required fields before build."""
        missing = []
        if self.embeddings is None:
            missing.append(
                "embeddings (call fetch_embedding_by_model_* / *_filename "
                "/ *_embedding_instance)"
            )

        if self.target_col not in self.data_df.columns:
            missing.append(
                "target_col '{}' (not found in dataset columns)".format(
                    self.target_col
                )
            )

        if self.persister_flag and self.model_short_name is None:
            missing.append(
                "model_short_name (required when persister is enabled)"
            )

        return missing

    def status(self) -> dict[str, object]:
        """Expose current builder state for easier interactive use."""
        return {
            "dataset_name": self.dataset.dataset_name,
            "task": self.task,
            "target_col": self.target_col,
            "split_type": self.split_type,
            "eval_all_splits": self.eval_all_splits,
            "has_embeddings": self.embeddings is not None,
            "model_short_name": self.model_short_name,
            "persister_enabled": self.persister_flag,
            "missing": self.validate(),
        }

    def build(self) -> LinearProbe:
        """Build LinearProbe instance.

        Returns:
            LinearProbe instance with set parameters.
        """
        missing = self.validate()
        if len(missing) > 0:
            raise ValueError(
                "Cannot build LinearProbe; missing configuration: {}".format(
                    "; ".join(missing)
                )
            )

        self.persister: LinearProbePersister | None = None

        if self.persister_flag:
            assert self.model_short_name is not None
            self.persister = LinearProbePersister(
                self.dataset,
                self.model_short_name,
                self.task,
                self.target_col,
                self.split_type
            )

        return LinearProbe(
            self.data_df,
            self.embeddings,
            self.target_col,
            self.task,
            self.splitter,
            self.evaluator,
            self.eval_all_splits,
            self.persister
        )
