import numpy as np

from filseg.config import AugmentConfig
from filseg.training.augment import augment_pair, dihedral, intensity_jitter


def test_augment_pair_keeps_input_and_target_aligned() -> None:
    rng = np.random.default_rng(0)
    y = np.zeros((2, 32, 32), np.float32)
    y[:, 3, 5] = 1
    x = y.copy()
    for _ in range(16):
        xa, ya = augment_pair(x, y, rng, AugmentConfig())
        assert xa.shape == ya.shape == (2, 32, 32)
        assert ya.sum() == 2
        hot = np.argwhere(ya[0] == 1)[0]
        assert xa[0][tuple(hot)] > xa[0].mean(), "input moved with the target"


def test_augment_pair_is_dihedral_then_jitter() -> None:
    """The random stream the released models were trained with: one dihedral draw, then the
    gain / offset draws."""
    x = np.random.default_rng(1).normal(size=(2, 16, 16)).astype(np.float32)
    cfg = AugmentConfig()
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    xa, _ = augment_pair(x, x, r1, cfg)
    xb, _ = dihedral(x, x, r2)
    xb = intensity_jitter(xb, r2, gain=cfg.gain, offset_std=cfg.offset_std)
    np.testing.assert_array_equal(xa, xb)
    assert r1.random() == r2.random()


def test_dihedral_off_only_jitters() -> None:
    cfg = AugmentConfig(dihedral=False, gain=(1.0, 1.0), offset_std=0.0)
    x = np.random.default_rng(2).normal(size=(2, 8, 8)).astype(np.float32)
    xa, ya = augment_pair(x, x.copy(), np.random.default_rng(0), cfg)
    np.testing.assert_array_equal(xa, x)
    np.testing.assert_array_equal(ya, x)
