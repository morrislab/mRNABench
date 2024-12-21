import os
import requests


def download_file(
    url: str,
    download_dir: str | None = None,
    force_redownload: bool = False
) -> tuple[str, bool]:
    """Download file at the given url.

    Args:
        url: URL of file to be downloaded.
        download_dir: Directory to store downloaded file.

    Returns:
        Path to downloaded file and whether file was downloaded.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    output_path = download_dir + "/" + os.path.basename(url)

    if os.path.isfile(output_path) and not force_redownload:
        print("File already downloaded.")
        return (output_path, False)

    # TODO: write progress bar.
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return (output_path, True)
