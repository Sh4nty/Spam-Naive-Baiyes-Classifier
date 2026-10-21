"""Download the raw corpora into data/raw/ (which is gitignored).

    python scripts/fetch_data.py

Enron-Spam: the single-file release of Metsis et al. (2006) by
MWiechmann/enron_spam_data, which keeps the arrival date from the original
file names.

SpamAssassin public corpus: the five official 2003/2005 tarballs from
spamassassin.apache.org; if that site cannot be reached, the copy in
github.com/stdlib-js/datasets-spam-assassin is cloned instead (same
messages, different folder names; the loader handles both).

Per the SpamAssassin corpus notice, never feed these messages into a live
mail system.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

ENRON_URL = (
    "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/"
    "master/enron_spam_data.zip"
)
ENRON_REPO = "https://github.com/MWiechmann/enron_spam_data"
SA_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SA_FILES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20050311_spam_2.tar.bz2",
]
SA_MIRROR = "https://github.com/stdlib-js/datasets-spam-assassin"


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "spam-nb-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def fetch_enron() -> None:
    dest = RAW / "enron_spam_data.zip"
    if dest.exists():
        print(f"Enron-Spam: already present at {dest}")
        return
    print("Enron-Spam:")
    try:
        download(ENRON_URL, dest)
    except Exception as error:
        print(f"  direct download failed ({error}); cloning the repository")
        dest.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", ENRON_REPO, tmp + "/repo"],
                check=True,
            )
            shutil.copy(Path(tmp) / "repo" / "enron_spam_data.zip", dest)


def _sa_official(dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name in SA_FILES:
            archive = Path(tmp) / name
            download(SA_BASE + name, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(dest, filter="data")


def _sa_mirror(dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        print(f"  cloning {SA_MIRROR}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", SA_MIRROR, tmp + "/repo"],
            check=True,
        )
        for folder in (Path(tmp) / "repo" / "data").iterdir():
            if folder.is_dir():
                shutil.copytree(folder, dest / folder.name)


def fetch_spamassassin() -> None:
    dest = RAW / "spamassassin"
    if dest.exists() and any(dest.iterdir()):
        print(f"SpamAssassin: already present at {dest}")
        return
    print("SpamAssassin:")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        _sa_official(dest)
    except Exception as error:  # site blocked or down: use the mirror
        print(f"  official site failed ({error}); using the GitHub mirror")
        shutil.rmtree(dest)
        dest.mkdir(parents=True)
        _sa_mirror(dest)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_enron()
    fetch_spamassassin()
    print(f"Done. Raw data is in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
