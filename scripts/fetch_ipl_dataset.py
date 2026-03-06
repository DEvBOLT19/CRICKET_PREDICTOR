"""Download the external IPL dataset into data/data.csv.

Usage:
    python scripts/fetch_ipl_dataset.py
"""

from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/12345k/IPL-Dataset/master/IPL/data.csv"
OUT_PATH = Path("data/data.csv")


def fetch_dataset(url: str = URL, out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=60) as response:
            content = response.read()
    except (URLError, HTTPError) as exc:
        raise RuntimeError(
            "Failed to download IPL dataset. "
            "Check network/proxy access to GitHub and retry."
        ) from exc

    if not content:
        raise RuntimeError("Downloaded file is empty.")

    out_path.write_bytes(content)
    print(f"Saved dataset to {out_path} ({len(content)} bytes)")


if __name__ == "__main__":
    fetch_dataset()
