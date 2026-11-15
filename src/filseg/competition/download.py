"""Competition data: download the archive through the Kaggle CLI, then extract it.

The zip is kept in ``data/raw``; the extracted folder is rebuilt from it.

Auth: the CLI reads ``~/.kaggle/access_token`` (or ``KAGGLE_API_TOKEN``). On a rented box, copy
the token file there; never commit it.
"""

import logging
import shutil
import subprocess
import zipfile

from filseg.infra.paths import COMPETITION, Paths

log = logging.getLogger(__name__)


def download_archive(paths: Paths, *, force: bool = False) -> None:
    """Download the competition zip into ``data/raw`` (idempotent)."""
    if paths.archive.exists() and not force:
        log.info("archive already present at %s", paths.archive)
        return
    paths.raw.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(paths.raw)]
    subprocess.run([*cmd, "--force"] if force else cmd, check=True)
    log.info("downloaded %s (%.0f MB)", paths.archive, paths.archive.stat().st_size / 2**20)


def extract_archive(paths: Paths) -> None:
    """Extract the zip into ``data/raw``, replacing any previous extraction."""
    shutil.rmtree(paths.dataset, ignore_errors=True)
    with zipfile.ZipFile(paths.archive) as zf:
        zf.extractall(paths.raw)
    n_train = len(list(paths.train_images.glob("*.jp*g")))
    n_test = len(list(paths.test_images.glob("*.jp*g")))
    log.info("extracted %d train / %d test images to %s", n_train, n_test, paths.dataset)
