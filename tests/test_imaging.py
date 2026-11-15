import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import blob
from filseg.config import CacheMode, Channel
from filseg.imaging import center_crop, paste_center
from filseg.preprocess import (
    GONG_DISK,
    Preprocessor,
    build_channels,
    disk_mask,
    locate_disk,
)


def _sun_with_halo(cx: float = 1023.1, cy: float = 1024.1, r: float = 900.4) -> np.ndarray:
    """GONG-like frame: limb-darkened disk, thin bright limb ring, and a halo that is brighter
    than the disk rim (which defeats intensity thresholds)."""
    yy, xx = np.ogrid[:2048, :2048]
    rho = np.hypot(xx - cx, yy - cy)
    mu = np.sqrt(np.clip(1 - (rho / (r - 2.5)) ** 2, 0, 1))
    img = np.where(blob(1024, 1024, 1010, 1010), 100.0, 0.0)  # halo, brighter than the rim
    img = np.where(rho < r - 2.5, 140 * (0.35 + 0.65 * mu), img)  # rim drops to ~50
    img = np.where(np.abs(rho - r) <= 1.5, 190.0, img)  # bright limb ring
    img[blob(int(cx - r - 30), int(cy), 40, 25)] = 160  # prominence on the limb
    img[blob(900, 900, 40, 40)] = 25  # sunspot: must not punch a hole in the disk
    return img.astype(np.uint8)


def test_disk_mask_is_the_limb_circle_despite_bright_halo() -> None:
    img = _sun_with_halo()
    d = locate_disk(img)
    assert abs(d.cx - 1023.1) < 0.5 and abs(d.cy - 1024.1) < 0.5 and abs(d.r - 900.4) < 1.0
    disk = disk_mask(img, margin=6)
    assert (disk == blob(1023, 1024, 894, 894)).mean() > 0.999
    assert disk[900, 900], "dark features inside the disk stay in the mask"
    assert not disk[1024, int(1023.1 - 900.4 - 30)], "prominences are not part of the disk"
    assert not disk[1024, 2000], "the halo is not part of the disk"


def test_locate_disk_follows_a_shifted_disk() -> None:
    d = locate_disk(_sun_with_halo(cx=1068.0, cy=1004.0, r=905.0))
    assert abs(d.cx - 1068.0) < 1.0 and abs(d.cy - 1004.0) < 1.0 and abs(d.r - 905.0) < 1.0


def test_locate_disk_falls_back_to_gong_geometry() -> None:
    assert locate_disk(np.zeros((2048, 2048), np.uint8)) == GONG_DISK


GOLDEN = Path(__file__).parent / "data" / "channels_golden.npz"


def test_channels_match_golden() -> None:
    """Every channel's output is frozen: the released models were trained on these values (the
    golden file was written by the training code). A deliberate change is a new `Channel`
    member; regenerate the file with `FILSEG_UPDATE_GOLDEN=1 uv run pytest tests/test_imaging.py`
    and check that the diff of the .npz only adds keys."""
    noise = np.random.default_rng(0).normal(0, 4, (2048, 2048))
    img = np.clip(_sun_with_halo() + noise, 0, 255).astype(np.uint8)
    x = build_channels(img, list(Channel))
    got = {}
    for ch, c in zip(Channel, x, strict=True):
        got[f"{ch}_grid"] = c[::32, ::32]
        got[f"{ch}_stats"] = np.array([c.mean(), c.std(), np.abs(c).max()], np.float64)
    if os.environ.get("FILSEG_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        np.savez_compressed(GOLDEN, **got)
        pytest.skip("golden values regenerated")
    want = np.load(GOLDEN)
    for key, ref in want.items():
        np.testing.assert_allclose(got[key], ref, atol=1e-4, err_msg=key)
    missing = sorted(set(got) - set(want))
    assert not missing, f"no golden values for {missing}: regenerate them"


def test_channels_shape_and_crop_roundtrip() -> None:
    x = build_channels(_sun_with_halo(), list(Channel))
    assert x.shape == (len(Channel), 2048, 2048) and x.dtype == np.float32
    assert not x[:, 0, 0].any(), "outside the disk is zeroed"
    crop = center_crop(x[0], 1888)
    assert (paste_center(crop)[80:-80, 80:-80] == x[0][80:-80, 80:-80]).all()


def test_preprocessor_cache_modes(tmp_path: Path) -> None:
    image = tmp_path / "img" / "sun.jpeg"
    image.parent.mkdir()
    cv2.imwrite(str(image), _sun_with_halo())
    channels, crop = [Channel.RAW, Channel.FLAT], 1888

    off = Preprocessor(channels, crop, cache_root=tmp_path / "cache", mode=CacheMode.OFF)
    ref = off(image)
    assert ref.shape == (2, crop, crop) and not (tmp_path / "cache").exists()

    ro = Preprocessor(channels, crop, cache_root=tmp_path / "cache", mode=CacheMode.READ)
    assert np.array_equal(ro(image), ref) and not (tmp_path / "cache").exists()

    rw = Preprocessor(channels, crop, cache_root=tmp_path / "cache", mode=CacheMode.READ_WRITE)
    assert np.array_equal(rw(image), ref)
    cached = rw.cache_path("sun")
    assert cached is not None and cached.exists()
    assert list(cached.parent.iterdir()) == [cached], "no temp files left behind"
    again = ro(image)
    assert isinstance(again, np.memmap) and np.array_equal(again, ref)
