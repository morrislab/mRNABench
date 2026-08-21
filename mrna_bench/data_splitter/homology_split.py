from collections import Counter
import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
import zipfile

import pandas as pd
import numpy as np

from mrna_bench.data_splitter.data_splitter import DataSplitter
from mrna_bench.utils import download_file, get_data_path


ENSEMBL_VERSION = 110
DEFAULT_SIMILARITY_THRESHOLD = 35
SPECIES_ALIASES = {
    "celegans": ("caenorhabditis_elegans", "celegans"),
    "chicken": ("gallus_gallus", "ggallus"),
    "chimpanzee": ("pan_troglodytes", "ptroglodytes"),
    "cow": ("bos_taurus", "btaurus"),
    "dog": ("canis_lupus_familiaris", "clfamiliaris"),
    "drosophila": ("drosophila_melanogaster", "dmelanogaster"),
    "human": ("homo_sapiens", "hsapiens"),
    "mouse": ("mus_musculus", "mmusculus"),
    "rat": ("rattus_norvegicus", "rnorvegicus"),
    "zebrafish": ("danio_rerio", "drerio"),
}


def _validate_ensembl_version(version: object) -> int:
    if isinstance(version, bool):
        raise ValueError("ensembl_version must be a positive integer.")
    if not isinstance(version, (int, np.integer)) or version <= 0:
        raise ValueError("ensembl_version must be a positive integer.")
    return int(version)


def _validate_similarity_threshold(threshold: object) -> float:
    if isinstance(threshold, bool):
        raise ValueError(
            "similarity_threshold must be a finite number from 0 to 100."
        )
    if not isinstance(
        threshold,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(
            "similarity_threshold must be a finite number from 0 to 100."
        )
    value = float(threshold)
    if not np.isfinite(value) or value < 0 or value > 100:
        raise ValueError(
            "similarity_threshold must be a finite number from 0 to 100."
        )
    return value


def _validate_test_size(test_size: object) -> float:
    if isinstance(test_size, bool) or not isinstance(
        test_size,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("test_size must be a finite number from 0 to 1.")
    value = float(test_size)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError("test_size must be a finite number from 0 to 1.")
    return value


def _resolve_ensembl_species(
    species: str,
) -> tuple[str, str]:
    if not species.replace("_", "").isalnum() or species.lower() != species:
        raise ValueError(
            "species must be a lowercase Ensembl name or known alias."
        )
    return SPECIES_ALIASES.get(species, (species, species))


def _write_csv(
    dataframe: pd.DataFrame,
    path: Path,
    sep: str = ",",
) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        dataframe.to_csv(temporary_path, sep=sep, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_homology_map(homology_df: pd.DataFrame) -> None:
    required_columns = {"gene_name", "gene_group"}
    if not required_columns.issubset(homology_df.columns):
        raise ValueError(
            "Homology map must contain gene_name and gene_group columns."
        )
    if homology_df[["gene_name", "gene_group"]].isna().any().any():
        raise ValueError("Homology map genes and groups cannot be null.")
    conflicting = homology_df.groupby(
        "gene_name"
    )["gene_group"].nunique()
    if (conflicting > 1).any():
        raise ValueError(
            "A gene cannot belong to multiple homology groups."
        )


def download_compara_paralogs(
    species: str,
    ensembl_version: int = ENSEMBL_VERSION,
) -> pd.DataFrame:
    """Download exact release-specific paralogs from Ensembl Compara.

    Args:
        species: Lowercase Ensembl species name or supported alias.
        ensembl_version: Ensembl release number to query.

    Returns:
        Paralog pairs and percent identities from Ensembl Compara.
    """
    import pymysql

    ensembl_version = _validate_ensembl_version(ensembl_version)
    species_name, _ = _resolve_ensembl_species(species)
    connection = pymysql.connect(
        host="ensembldb.ensembl.org",
        user="anonymous",
        port=5306,
        database="ensembl_compara_{}".format(ensembl_version),
        connect_timeout=30,
        read_timeout=900,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT mlss.method_link_species_set_id
                FROM method_link_species_set mlss
                JOIN species_set ss
                  ON ss.species_set_id = mlss.species_set_id
                JOIN genome_db gd ON gd.genome_db_id = ss.genome_db_id
                WHERE mlss.method_link_id = 202
                  AND gd.name = %s
                  AND gd.genome_component IS NULL
                """,
                (species_name,),
            )
            rows = cursor.fetchall()
            if len(rows) != 1:
                raise ValueError(
                    "Expected one Ensembl paralog set for {}, "
                    "found {}.".format(
                        species, len(rows)
                    )
                )
            cursor.execute(
                """
                SELECT
                    gm1.display_label,
                    gm2.display_label,
                    hm1.perc_id
                FROM homology h FORCE INDEX (method_link_species_set_id)
                JOIN homology_member hm1
                  ON hm1.homology_id = h.homology_id
                JOIN homology_member hm2
                  ON hm2.homology_id = h.homology_id
                 AND hm2.gene_member_id <> hm1.gene_member_id
                JOIN gene_member gm1
                  ON gm1.gene_member_id = hm1.gene_member_id
                JOIN gene_member gm2
                  ON gm2.gene_member_id = hm2.gene_member_id
                WHERE h.method_link_species_set_id = %s
                  AND gm1.display_label IS NOT NULL
                  AND gm2.display_label IS NOT NULL
                """,
                (rows[0][0],),
            )
            paralogs = pd.DataFrame(cursor.fetchall(), columns=[
                "Gene name",
                "Paralogue associated gene name",
                "Paralogue %id. target gene identical to query gene",
            ])
            return (
                paralogs.sort_values(
                    list(paralogs.columns),
                    ascending=False,
                )
                .drop_duplicates(paralogs.columns[:2].tolist())
                .reset_index(drop=True)
            )
    finally:
        connection.close()


def build_homology_map(
    paralogs: pd.DataFrame,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """Reproduce the published greedy Ensembl paralog grouping.

    Args:
        paralogs: Ensembl paralog export containing gene names, associated
            paralogs, and percent identities.
        similarity_threshold: Minimum percent identity for grouping genes.

    Returns:
        Dataframe mapping each gene name to one homology group.
    """
    similarity_threshold = _validate_similarity_threshold(
        similarity_threshold
    )
    gene_col = "Gene name"
    paralog_cols = [
        column for column in paralogs
        if "paralogue associated gene name" in column.lower()
    ]
    identity_cols = [
        column for column in paralogs
        if "paralogue %id." in column.lower()
    ]
    if gene_col not in paralogs or len(paralog_cols) != 1:
        raise ValueError(
            "Paralog table must contain 'Gene name' and one paralogue "
            "associated gene-name column."
        )
    if len(identity_cols) != 1:
        raise ValueError(
            "Paralog table must contain one percent-identity column."
        )

    paralog_col = paralog_cols[0]
    identity_col = identity_cols[0]
    pairs = paralogs.rename(columns={
        gene_col: "gene_symbol",
        paralog_col: "gene_symbol2",
        identity_col: "identity",
    }).copy()
    identity = pd.to_numeric(pairs["identity"], errors="coerce")
    if (pairs["identity"].notna() & identity.isna()).any():
        raise ValueError(
            "Paralog percent identities must be numeric when provided."
        )
    pairs["identity"] = identity
    pairs = (
        pairs.sort_values(
            ["gene_symbol2", "gene_symbol", "identity"],
            ascending=False,
        )
        .drop_duplicates(["gene_symbol", "gene_symbol2"])
        .dropna(subset=["gene_symbol", "gene_symbol2"])
    )
    pairs = pairs[pairs["identity"] > similarity_threshold]

    gene_groups: dict[str, int] = {}
    conflicts: list[tuple[int, int]] = []
    next_group = 0
    for gene, paralog in pairs[
        ["gene_symbol", "gene_symbol2"]
    ].itertuples(index=False, name=None):
        gene_group = gene_groups.get(gene)
        paralog_group = gene_groups.get(paralog)
        if gene_group is None and paralog_group is None:
            gene_groups[gene] = gene_groups[paralog] = next_group
            next_group += 1
        elif gene_group is None and paralog_group is not None:
            gene_groups[gene] = paralog_group
        elif gene_group is not None and paralog_group is None:
            gene_groups[paralog] = gene_group
        else:
            assert gene_group is not None and paralog_group is not None
            if gene_group != paralog_group:
                conflicts.append((
                    min(gene_group, paralog_group),
                    max(gene_group, paralog_group),
                ))

    homology = pd.DataFrame(
        gene_groups.items(),
        columns=["gene_name", "gene_group"],
    )
    group_sizes = homology.groupby("gene_group").size().to_dict()
    # Merge groups when conflicts cover at least half their cross-product.
    parent = {group: group for group in group_sizes}

    def root(group: int) -> int:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    for (group1, group2), count in Counter(conflicts).most_common():
        if count >= group_sizes[group1] * group_sizes[group2] * 0.5:
            parent[root(group2)] = root(group1)
    homology["gene_group"] = homology["gene_group"].map(root)

    final_groups = homology.set_index("gene_name")["gene_group"].to_dict()
    incident_edges: Counter[int] = Counter()
    # Connectivity counts every thresholded pair touching a group.
    for gene, paralog in pairs[
        ["gene_symbol", "gene_symbol2"]
    ].itertuples(index=False, name=None):
        incident_edges.update({final_groups[gene], final_groups[paralog]})

    group_sizes = homology.groupby("gene_group").size()
    connectedness = {
        group: 1.0
        if size == 1
        else incident_edges[group] / (size * (size - 1))
        for group, size in group_sizes.items()
    }
    homology["perc_connected"] = homology["gene_group"].map(connectedness)
    homology["n_members"] = (
        homology["gene_group"].map(group_sizes).astype(float)
    )
    return homology[homology["perc_connected"] > 0.15]


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
    test_size = _validate_test_size(test_size)
    _validate_homology_map(homology_df)
    # Map genes to their respective homology groups
    homo_group_map = homology_df.set_index("gene_name")["gene_group"].to_dict()

    gene_groups = np.array([
        ("group", homo_group_map[gene])
        if gene in homo_group_map else ("gene", gene)
        for gene in genes
    ], dtype=object)

    group_to_index: dict[tuple[str, object], list[int]] = {}

    # Populate the group_to_index dictionary
    for i, group in enumerate(gene_groups):
        group_to_index.setdefault(tuple(group), []).append(i)

    gene_index = np.arange(len(genes))

    np.random.RandomState(random_state).shuffle(gene_index)

    len_of_train = int(len(gene_index) * (1 - test_size))

    train_indices: list[int] = []
    test_indices: list[int] = []

    seen_groups = set()

    # Whole groups may overshoot; add balancing if ratio precision matters
    # more than this simple leakage-safe assignment.
    for index in gene_index:
        group = tuple(gene_groups[index])
        if group in seen_groups:
            continue

        seen_groups.add(group)

        if len(train_indices) < len_of_train:
            train_indices.extend(group_to_index[group])
        else:
            test_indices.extend(group_to_index[group])

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
        default_split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs
    ):
        """Initialize HomologySplitter.

        Homology splitting requires a dataframe mapping genes to a group
        of homologous genes. This dataframe must have columns "gene_name" and
        "gene_group". Published maps are materialized one species at a time.

        Maps are cached as:
        ensembl-{version}/sim-{threshold}pct/{species}.csv

        Args:
            default_split_ratio: Ratio of training, validation, test splits.
            **kwargs: Homology-specific arguments.
        """
        super().__init__(default_split_ratio)

        species = kwargs["species"]
        if not isinstance(species, str):
            raise TypeError("species must be a string.")
        self.species = species
        _, cache_name = _resolve_ensembl_species(species)

        homology_map_path: str | None = kwargs.get("homology_map_path")
        homology_source_path: str | None = kwargs.get("homology_source_path")
        similarity_threshold: float | None = kwargs.get(
            "similarity_threshold"
        )
        ensembl_version = _validate_ensembl_version(
            kwargs.get("ensembl_version", ENSEMBL_VERSION)
        )
        force_redownload: bool = kwargs.get("force_redownload", False)
        on_the_fly = any((
            homology_source_path is not None,
            similarity_threshold is not None,
            ensembl_version != ENSEMBL_VERSION,
            species not in SPECIES_ALIASES,
        ))
        threshold = _validate_similarity_threshold(
            DEFAULT_SIMILARITY_THRESHOLD
            if similarity_threshold is None
            else similarity_threshold
        )

        if homology_map_path is None:
            self.homology_map_path = get_data_path() + "/homology_maps"
        else:
            self.homology_map_path = homology_map_path
        release_dir = Path(
            self.homology_map_path,
            "ensembl-{}".format(ensembl_version),
        )
        if homology_source_path is not None:
            source_hash = hashlib.sha256()
            with open(homology_source_path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    source_hash.update(chunk)
            release_dir /= "custom-{}".format(
                source_hash.hexdigest()[:12]
            )
        elif on_the_fly:
            release_dir /= "generated"
        map_dir = release_dir / "sim-{:g}pct".format(threshold)
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / "{}.csv".format(cache_name)
        source_path = release_dir / "{}.tsv".format(cache_name)

        if on_the_fly:
            if map_path.exists() and not force_redownload:
                self.homology_df = pd.read_csv(map_path)
            else:
                if homology_source_path is None:
                    if source_path.exists() and not force_redownload:
                        paralogs = pd.read_csv(source_path, sep="\t")
                    else:
                        paralogs = download_compara_paralogs(
                            self.species,
                            ensembl_version,
                        )
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        _write_csv(paralogs, source_path, sep="\t")
                else:
                    paralogs = pd.read_csv(
                        homology_source_path,
                        sep=None,
                        engine="python",
                    )
                self.homology_df = build_homology_map(
                    paralogs,
                    similarity_threshold=threshold,
                )
                _write_csv(self.homology_df, map_path)
        else:
            map_path = self._ensure_published_map(
                map_path,
                force_redownload,
            )
            self.homology_df = pd.read_csv(map_path)

        _validate_homology_map(self.homology_df)

    def _ensure_published_map(
        self,
        map_path: Path,
        force_redownload: bool,
    ) -> Path:
        """Materialize the requested published species map."""
        if map_path.exists() and not force_redownload:
            return map_path

        archive = Path(
            self.homology_map_path,
            Path(self.HOMO_URL).name,
        )
        if force_redownload or not zipfile.is_zipfile(archive):
            with TemporaryDirectory(
                dir=self.homology_map_path
            ) as temporary_dir:
                downloaded = Path(download_file(
                    self.HOMO_URL,
                    temporary_dir,
                    True,
                ))
                if not zipfile.is_zipfile(downloaded):
                    raise zipfile.BadZipFile(
                        "Downloaded homology archive is invalid."
                    )
                downloaded.replace(archive)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            files = {
                Path(name).name: name for name in zip_ref.namelist()
            }
            filename = "{}_homology_map.csv".format(self.species)
            if filename not in files:
                raise ValueError(
                    "Published homology map is unavailable for {}.".format(
                        self.species
                    )
                )
            with NamedTemporaryFile(
                dir=map_path.parent,
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
            try:
                with (
                    zip_ref.open(files[filename]) as source,
                    open(temporary_path, "wb") as target,
                ):
                    shutil.copyfileobj(source, target)
                temporary_path.replace(map_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return map_path

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
        if df["gene"].isna().any():
            raise ValueError("Gene cannot be null for homology split.")

        genes = df["gene"].to_list()

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
