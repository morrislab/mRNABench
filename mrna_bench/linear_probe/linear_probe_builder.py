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

        # Set default values
        self.target_col = "target"

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

    def build(self) -> LinearProbe:
        """Build LinearProbe instance.

        Returns:
            LinearProbe instance with set parameters.
        """
        self.persister: LinearProbePersister | None = None

        if hasattr(self, "persister_flag") and self.persister_flag:
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
