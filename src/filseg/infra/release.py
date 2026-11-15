"""Model weights as GitHub release assets: pack them on the training box, fetch them anywhere.

``pack`` collects what inference needs for one config into a flat directory: the best checkpoint
of every fold of each model experiment (the config itself, or each ``blend`` member), their
resolved ``config.yaml``, and the config's inference-time post-processing files
(``post_params.json``, ``blend.json``, the scorer's ``params.json`` and ``model*.txt``; the
crossfit ``fold<k>.txt`` models are training-only and left out). A
``manifest.json`` maps each asset to its path under ``outputs/`` with its SHA-256 and size.
Upload the directory as the assets of one release (``gh release create <tag> dist/weights/*``).

``fetch`` reads ``manifest.json`` from a release (or any base URL, ``file://`` included),
downloads each asset into ``outputs/`` at its recorded path and verifies its hash, so the usual
commands run unchanged afterwards. Files already present with the right hash are skipped.
"""

import hashlib
import logging
import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from filseg.config import ExperimentConfig, ScorerMode
from filseg.infra.paths import Paths

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
DEFAULT_TAG = "weights-v1"
_CHUNK = 1 << 20


class Asset(BaseModel):
    asset: str  # flat file name in the release
    path: str  # destination relative to the output root, POSIX separators
    sha256: str
    size: int


class Manifest(BaseModel):
    config: str
    files: list[Asset]


def release_url(repo: str, tag: str = DEFAULT_TAG) -> str:
    """Base URL of the assets of release ``tag`` of GitHub repo ``owner/name``."""
    return f"https://github.com/{repo}/releases/download/{tag}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def release_files(cfg: ExperimentConfig, paths: Paths) -> list[Path]:
    """Files under ``paths.output_root`` that inference with ``cfg`` reads; raises if a
    required one is missing (a fold checkpoint, the scorer)."""
    exp = paths.experiment(cfg.name)
    files: list[Path] = []
    for name in cfg.blend or [cfg.name]:
        member = paths.experiment(name)
        for k in cfg.train_folds:
            ckpt = member.model(k)
            if not ckpt.exists():
                raise FileNotFoundError(f"{name}: no checkpoint for fold {k} in {member.fold(k)}")
            files.append(ckpt)
        if member.config.exists():
            files.append(member.config)
    files += [p for p in (exp.post_params, exp.blend) if p.exists() and p not in files]
    if cfg.scorer.mode != ScorerMode.OFF:
        scorer = exp.scorer
        if not scorer.params.exists() or not scorer.model.exists():
            raise FileNotFoundError(f"no fitted scorer in {scorer.dir} (run fit-scorer)")
        files += [scorer.params, *sorted(scorer.dir.glob("model*.txt"))]
    return files


def pack(cfg: ExperimentConfig, paths: Paths, out: Path) -> Manifest:
    """Copy `release_files` into ``out`` under flat asset names and write ``manifest.json``."""
    out.mkdir(parents=True, exist_ok=True)
    assets = []
    for src in release_files(cfg, paths):
        rel = src.relative_to(paths.output_root).as_posix()
        name = rel.replace("/", "--")
        shutil.copy2(src, out / name)
        assets.append(Asset(asset=name, path=rel, sha256=sha256(src), size=src.stat().st_size))
        log.info("packed %s", rel)
    manifest = Manifest(config=cfg.name, files=assets)
    (out / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n")
    total = sum(a.size for a in assets) / 1e6
    log.info("wrote %d assets (%.0f MB) and %s to %s", len(assets), total, MANIFEST, out)
    return manifest


def fetch(base_url: str, paths: Paths, *, force: bool = False) -> list[Path]:
    """Download the release at ``base_url`` into ``paths.output_root``; returns the files."""
    base = base_url.rstrip("/")
    manifest = Manifest.model_validate_json(_read(f"{base}/{MANIFEST}"))
    root = paths.output_root.resolve()
    out = []
    for a in manifest.files:
        dest = (root / a.path).resolve()
        if not dest.is_relative_to(root):  # a manifest must not write outside outputs/
            raise ValueError(f"manifest path {a.path!r} leaves the output root")
        fresh = dest.exists() and dest.stat().st_size == a.size and sha256(dest) == a.sha256
        if fresh and not force:
            log.info("up to date: %s", a.path)
            out.append(dest)
            continue
        _download(f"{base}/{a.asset}", dest, a)
        out.append(dest)
    log.info("fetched %d files of %s into %s", len(out), manifest.config, root)
    return out


def _read(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        data: bytes = resp.read()
    return data


def _chunks(url: str) -> Iterator[bytes]:
    with urllib.request.urlopen(url) as resp:
        while chunk := resp.read(_CHUNK):
            yield chunk


def _download(url: str, dest: Path, a: Asset) -> None:
    """Stream ``url`` to ``dest.part``, check size and hash, then move it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    n = 0
    with (
        tmp.open("wb") as f,
        tqdm(total=a.size, unit="B", unit_scale=True, desc=a.asset, leave=False) as bar,
    ):
        for chunk in _chunks(url):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
            bar.update(len(chunk))
    if n != a.size or h.hexdigest() != a.sha256:
        tmp.unlink()
        raise OSError(
            f"{a.asset}: got {n} bytes / sha256 {h.hexdigest()}, manifest says "
            f"{a.size} / {a.sha256}"
        )
    tmp.replace(dest)
    log.info("downloaded %s", a.path)
