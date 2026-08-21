import hashlib

import pytest
import requests

from mrna_bench.utils import download_file


class Response:
    """Minimal streaming response used by download tests."""

    def __init__(self, chunks):
        """Initialize a response with chunks or injected exceptions."""
        self.chunks = chunks
        self.headers = {
            "content-length": str(
                sum(len(chunk) for chunk in chunks if isinstance(chunk, bytes))
            )
        }

    def __enter__(self):
        """Enter the response context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context without suppressing exceptions."""
        return False

    def raise_for_status(self):
        """Model a successful HTTP status."""

    def iter_content(self, chunk_size):
        """Yield chunks or raise an injected exception."""
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def test_download_file_replaces_invalid_cached_file(tmp_path, monkeypatch):
    """A checksum mismatch triggers an atomic replacement download."""
    destination = tmp_path / "source.bin"
    destination.write_bytes(b"invalid")
    content = b"complete source"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, stream: Response([content]),
    )

    result = download_file(
        "https://example.com/source.bin",
        str(tmp_path),
        expected_sha256=checksum,
    )

    assert result == str(destination)
    assert destination.read_bytes() == content
    assert not (tmp_path / "source.bin.part").exists()


def test_download_file_cleans_interrupted_partial(tmp_path, monkeypatch):
    """An interrupted transfer leaves no partial final download."""
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, stream: Response([
            b"partial",
            requests.ConnectionError("interrupted"),
        ]),
    )

    with pytest.raises(requests.ConnectionError, match="interrupted"):
        download_file(
            "https://example.com/source.bin",
            str(tmp_path),
        )

    assert not (tmp_path / "source.bin").exists()
    assert not (tmp_path / "source.bin.part").exists()
