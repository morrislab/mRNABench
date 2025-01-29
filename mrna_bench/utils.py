import os
import pathlib
import requests
from tqdm import tqdm
import yaml


def download_file(
    url: str,
    download_dir: str,
    force_redownload: bool = False
) -> tuple[str, bool]:
    """Download file at the given url.

    Args:
        url: URL of file to be downloaded.
        download_dir: Directory to store downloaded file.
        force_redownload: Forces download even if file already exists.

    Returns:
        Path to downloaded file and whether file was downloaded.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))

    output_path = download_dir + "/" + os.path.basename(url)

    if os.path.isfile(output_path) and not force_redownload:
        print("File already downloaded.")
        return (output_path, False)

    with open(output_path, "wb") as f:
        with tqdm(total=total_bytes, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    return (output_path, True)


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
            pathlib.Path(data_dir_path).mkdir(parents=True, exist_ok=True)

    def get_data_path(self) -> str:
        """Load data_dir_path from config.

        Throws exception if config is not yet initialized.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path) as stream:
                return yaml.safe_load(stream)["data_path"]
        else:
            raise RuntimeError((
                "Data storage path is not set. Please run: "
                "mrna_bench.update_data_path(path_to_store_data)"
            ))


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


def get_model_weights_path() -> str:
    """Get path where model weights are stored.

    Returns:
        Directory where model weights are stored.
    """
    dm = DataManager()
    data_path = pathlib.Path(dm.get_data_path())
    model_path = data_path / "model_weights"

    return str(model_path)
