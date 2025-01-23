from pathlib import Path
import shutil
import zipfile

import pandas as pd
import numpy as np

from mrna_bench import get_data_path
from mrna_bench.data_splitter.data_splitter import DataSplitter
from mrna_bench.utils import download_file


def train_test_split_homologous(
    genes: list[str],
    homology_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int | None = None
) -> tuple[list[int], list[int]]:
    """Split genes into two sets with homologous genes in the same set.

    Args:
        genes: List of gene names.
        homology_df: DataFrame with columns "gene_name" and "gene_group".
        test_size: Fraction of data allocated to test split.
        random_state: Defaults to None.

    Returns:
        Dictionary with keys "train_indices" and "test_indices" containing the
        indices of the genes in the train and test sets respectively.
    """
    # Map genes to their respective homology groups
    homo_group_map = homology_df.set_index("gene_name")["gene_group"].to_dict()

    gene_groups = np.array([homo_group_map.get(gene, None) for gene in genes])

    group_to_index: dict[int, list[int]] = {}

    # Populate the group_to_index dictionary
    for i, group in enumerate(gene_groups):
        if group is not None:
            group_to_index.setdefault(group, []).append(i)

    gene_index = np.arange(len(genes))

    np.random.seed(random_state)
    np.random.shuffle(gene_index)

    len_of_train = int(len(gene_index) * (1 - test_size))

    train_indices: list[int] = []
    test_indices: list[int] = []

    seen_groups = set()

    for index in gene_index:
        group = gene_groups[index]
        if group is not None and group in seen_groups:
            continue

        seen_groups.add(group)

        if len(train_indices) < len_of_train:
            train_indices.extend(group_to_index.get(group, [index]))
        else:
            test_indices.extend(group_to_index.get(group, [index]))

    train_indices = [int(ind) for ind in train_indices]
    test_indices = [int(ind) for ind in test_indices]

    return train_indices, test_indices


class HomologySplitter(DataSplitter):
    """Homology-based data splitter.

    Uses an external homology mapping file to construct train / test splits.
    Genes which are homologous are kept within the same 'side' of the data
    split to reduce data leakage.
    """

    HOMO_URL = (
        "https://zenodo.org/records/13910050/files/"
        "homology_maps_homologene.zip"
    )

    def __init__(
        self,
        species: list[str],
        homology_map_path: str | None = None,
        force_redownload: bool = False
    ):
        """Initialize HomologySplitter.

        Homology splitting requires a dataframe mapping genes to a group
        of homologous genes. This dataframe must have columns "gene_name" and
        "gene_group". If homology_map_path is not specified, relationships
        processed from homologene will be downloaded.

        Homology map files should be named as: {species}_homology_map.csv

        Args:
            species: List of species that genes are from.
            homology_map_path: Path to homology maps.
            force_redownload: Forces redownload of homology maps.
        """
        super().__init__()

        if homology_map_path is None:
            data_storage_path = get_data_path()
            self.homology_map_path = data_storage_path + "/homology_maps"

            if not Path(self.homology_map_path).exists():
                Path(self.homology_map_path).mkdir(exist_ok=True)
                force_redownload = True
        else:
            self.homology_map_path = homology_map_path

        if force_redownload:
            out = download_file(self.HOMO_URL, self.homology_map_path, True)[0]

            with zipfile.ZipFile(out, "r") as zip_ref:
                for file in zip_ref.namelist():
                    if not file.endswith("/"):
                        source = zip_ref.open(file)
                        path = Path(self.homology_map_path) / Path(file).name

                        with open(path, "wb") as target:
                            shutil.copyfileobj(source, target)

            Path(out).unlink()

        self.species = species
        self.homology_df = self.get_homology_df()

    def get_homology_df(self) -> pd.DataFrame:
        """Get homology dataframe for specified species.

        If multiple species exists, naively merges dataframes, merging homology
        groups if gene name overlap exists.

        Returns:
            Dataframe with gene names mapped to homology groups.
        """
        species_maps = []
        for s in self.species:
            homology_dir = Path(self.homology_map_path)
            homology_map = homology_dir / "{}_homology_map.csv".format(s)
            if not homology_map.exists():
                print("Homology map for species {} is missing.".format(s))
            else:
                species_maps.append(pd.read_csv(homology_map))

        max_groups = [max(s_df["gene_group"]) for s_df in species_maps]
        cusum_max_groups = np.cumsum(max_groups)

        for i, s_df in enumerate(species_maps[1:]):
            s_df["gene_group"] += cusum_max_groups[i]

        out_df = pd.concat(species_maps, axis=0, ignore_index=True)

        # Merge identical gene names
        out_df.groupby("gene_name", as_index=False).min()
        out_df.reset_index(drop=True, inplace=True)

        return out_df

    def split_df(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe rows into train and test df using homology split.

        Args:
            df: Dataframe to split.
            test_size: Fraction of dataset to assign to test split.
            random_seed: Random seed used for sampling rows during splitting.

        Returns:
            Dataframe containing train data and test data.
        """
        if "gene" not in df.columns:
            raise ValueError("Gene must be column for homology split.")

        genes = df["gene"]

        train_indices, test_indices = train_test_split_homologous(
            genes,
            self.homology_df,
            test_size,
            random_seed
        )

        assert len(set(train_indices).intersection(set(test_indices))) == 0

        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]

        return train_df, test_df
