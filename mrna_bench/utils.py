import os
from pathlib import Path
import requests
from tqdm import tqdm
import yaml

from warnings import warn


def download_file(
    url: str,
    download_dir: str,
    force_redownload: bool = False
) -> str:
    """Download file at the given url.

    Args:
        url: URL of file to be downloaded.
        download_dir: Directory to store downloaded file.
        force_redownload: Forces download even if file already exists.

    Returns:
        Path to downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))

    output_path = download_dir + "/" + os.path.basename(url)

    if os.path.isfile(output_path) and not force_redownload:
        print("File already downloaded.")
        return output_path

    with open(output_path, "wb") as f:
        with tqdm(total=total_bytes, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    return output_path


class DataManager:
    """Helper class tracking the directory where benchmark data is stored."""

    def __init__(self):
        """Initialize DataManager."""
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = self.curr_dir + "/config.yaml"

    def update_data_path(self, data_dir_path: str):
        """Update path to data storage directory.

        Args:
            data_dir_path: New path to data storage directory.
        """
        data_dir_path = os.path.normpath(data_dir_path)
        data_dir_path = str(Path(data_dir_path).expanduser())

        config = {"data_path": data_dir_path}

        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            data["data_path"] = data_dir_path

            with open(self.config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        if not os.path.exists(data_dir_path):
            print("Specified data path does not exist. Making directories.")
            Path(data_dir_path).mkdir(parents=True, exist_ok=True)

    def get_data_path(self) -> str:
        """Load data_dir_path from config.

        Throws exception if config is not yet initialized.

        Returns:
            Path to data storage directory.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path) as stream:
                return yaml.safe_load(stream)["data_path"]
        else:
            raise RuntimeError((
                "Data storage path is not set. Please run: "
                "mrna_bench.update_data_path(path_to_store_data)"
            ))

    def update_model_weights_path(self, model_weights_path: str):
        """Update path to model weights storage directory.

        Args:
            model_weights_path: New path to model weights storage directory.
        """
        model_weights_path = os.path.normpath(model_weights_path)
        model_weights_path = str(Path(model_weights_path).expanduser())

        config = {"model_weights_path": model_weights_path}

        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            data["model_weights_path"] = model_weights_path

            with open(self.config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        if not os.path.exists(model_weights_path):
            print("Specified weights path does not exist. Making directories.")
            Path(model_weights_path).mkdir(parents=True, exist_ok=True)

    def get_model_weights_path(self) -> str:
        """Load model_weights_path from config.

        Returns default path if config is not yet initialized, which uses the
        data path as the root.

        Returns:
            Path to model weights storage directory.
        """
        c_init = os.path.exists(self.config_path)
        p_init = "model_weights_path" in yaml.safe_load(open(self.config_path))

        if not c_init or not p_init:
            warn("Model weights storage path is not set. Using default path.")
            data_path = Path(self.get_data_path())
            self.update_model_weights_path(str(data_path / "model_weights"))

        with open(self.config_path) as stream:
            return yaml.safe_load(stream)["model_weights_path"]


def update_data_path(path_to_data: str):
    """Update path to benchmark data storage directory.

    Args:
        path_to_data: New path to directory where data is stored.
    """
    dm = DataManager()
    dm.update_data_path(path_to_data)


def get_data_path() -> str:
    """Get path where benchmark data is stored.

    Returns:
        Directory where benchmark data is stored.
    """
    dm = DataManager()
    return dm.get_data_path()


def update_model_weights_path(path_to_weights: str):
    """Update path to model weights storage directory.

    Args:
        path_to_weights: New path to directory where model weights are stored.
    """
    dm = DataManager()
    dm.update_model_weights_path(path_to_weights)


def get_model_weights_path() -> str:
    """Get path where model weights are stored.

    Returns:
        Directory where model weights are stored.
    """
    dm = DataManager()
    return dm.get_model_weights_path()


def set_model_cache_var(cache_var: str = "HF_HUB_CACHE") -> str | None:
    """Set the HF_HUB_CACHE environment variable.

    This function sets the HF_HUB_CACHE environment variable to the
    directory where model weights are stored. This is necessary for
    downloading models from Hugging Face Hub.

    Args:
        cache_var: The name of the cache variable to set.

    Returns:
        The path to the previously set cache_var directory.
    """
    old_value = os.environ.get(cache_var)
    new_value = os.path.join(get_model_weights_path(), "")
    os.environ[cache_var] = new_value
    return old_value


def revert_model_cache_var(
    old_value: str | None,
    cache_var: str = "HF_HUB_CACHE"
):
    """Reverts HF_HUB_CACHE to the old value.

    Restores the HF_HUB_CACHE environment variable to its previous
    value. If the old value is None, the variable is removed.

    Args:
        old_value: The previous value of HF_HUB_CACHE.
    """
    if old_value is None:
        os.environ.pop(cache_var, None)
    else:
        os.environ[cache_var] = old_value
