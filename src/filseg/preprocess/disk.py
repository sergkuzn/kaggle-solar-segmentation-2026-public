"""Solar-disk geometry: limb fit (`locate_disk`) and the disk mask."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from filseg.competition.coco import IMAGE_SIZE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Disk:
    """Solar-disk circle in pixel coordinates."""

    cx: float
    cy: float
    r: float


# GONG frames are re-centred and rescaled to a fixed disk: `locate_disk` over all 887 train+test
# frames gives centre (1023.1 +- 0.4, 1024.1 +- 0.4) and radius 900.4 +- 0.7 px (the radius of
# the bright limb ring, which spans r-2.5 .. r+4 with a dark processing dip at r-5 .. r-3).
GONG_DISK = Disk(1023.1, 1024.1, 900.4)
# Limb searches around the prior centre, as (inner, outer) radius offsets: the narrow one covers
# normal frames, the wide one a frame whose disk is shifted (stops short of the field-of-view
# edge at ~1024 px, which is another strong circular edge).
_SEARCH_WINDOWS = ((-30.0, 30.0), (-150.0, 60.0))
_RADIUS_TOL = 0.05  # accepted fitted radius: prior radius +- 5%
_MIN_COVERAGE = 0.4  # min fraction of rays whose limb edge lies on the fitted circle
# Min ridge strength of a limb point: the limb ring scores >= ~2.0 on 95% of rays of every frame,
# the strongest ridges inside the disk <= ~1.2.
_MIN_RIDGE = 1.5


def _prior(shape: tuple[int, ...]) -> Disk:
    s = shape[1] / IMAGE_SIZE
    return Disk(GONG_DISK.cx * s, GONG_DISK.cy * s, GONG_DISK.r * s)


def _limb_edges(
    img: np.ndarray, centre: Disk, r_lo: float, r_hi: float, n_rays: int
) -> tuple[np.ndarray, np.ndarray]:
    """Strongest limb ridge on each of ``n_rays`` rays from ``centre`` within [r_lo, r_hi);
    rays without a ridge of at least `_MIN_RIDGE` are dropped."""
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = np.arange(r_lo, r_hi, 0.5)
    xs = (centre.cx + np.outer(np.cos(th), r)).astype(np.float32)
    ys = (centre.cy + np.outer(np.sin(th), r)).astype(np.float32)
    prof = cv2.remap(img.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderValue=0.0)
    # GONG frames carry a thin bright ring at the limb: a ridge detector (-d2/dr2 at ~2 px scale)
    # locks onto it more consistently than a step-edge one (fit spread 0.7 px vs 1.3 px in r)
    ridge = -gaussian_filter1d(prof, 4.0, axis=1, order=2)
    j = ridge.argmax(axis=1)
    i = np.arange(n_rays)
    keep = ridge[i, j] >= _MIN_RIDGE
    return xs[i, j][keep].astype(np.float64), ys[i, j][keep].astype(np.float64)


def _circle_lsq(x: np.ndarray, y: np.ndarray) -> Disk:
    """Algebraic least-squares circle through points."""
    a = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    return Disk(float(cx), float(cy), float(np.sqrt(c + cx**2 + cy**2)))


def _ransac_circle(
    x: np.ndarray,
    y: np.ndarray,
    r_range: tuple[float, float],
    *,
    tol: float = 2.0,
    iters: int = 256,
) -> tuple[Disk, int] | None:
    """Robust circle fit (prominences, spots and halo edges are outliers) -> (circle, number of
    inliers), or None when no 3-point sample gives a radius in ``r_range``."""
    if len(x) < 3:
        return None
    idx = np.random.default_rng(0).integers(0, len(x), size=(iters, 3))
    (ax, bx, qx), (ay, by, qy) = x[idx].T, y[idx].T
    d = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
    ok = np.abs(d) > 1e-6
    d = np.where(ok, d, 1.0)
    a2, b2, q2 = ax**2 + ay**2, bx**2 + by**2, qx**2 + qy**2
    ux = (a2 * (by - qy) + b2 * (qy - ay) + q2 * (ay - by)) / d
    uy = (a2 * (qx - bx) + b2 * (ax - qx) + q2 * (bx - ax)) / d
    ur = np.hypot(ax - ux, ay - uy)
    ok &= (ur >= r_range[0]) & (ur <= r_range[1])
    if not ok.any():
        return None
    ux, uy, ur = ux[ok, None], uy[ok, None], ur[ok, None]
    inliers = np.abs(np.hypot(x - ux, y - uy) - ur) < tol
    best = inliers[inliers.sum(axis=1).argmax()]
    fit = _circle_lsq(x[best], y[best])
    inl = np.abs(np.hypot(x - fit.cx, y - fit.cy) - fit.r) < tol
    if inl.sum() >= 3:
        fit = _circle_lsq(x[inl], y[inl])
    return fit, int(inl.sum())


def locate_disk(img: np.ndarray, *, n_rays: int = 360) -> Disk:
    """Solar-disk circle of a frame: a limb fit, falling back to the fixed GONG geometry.

    Intensity thresholds cannot find the limb: limb darkening drops the disk rim to ~55-80
    while the halo just outside is often ~90-110, so they leak into the halo (and grab bright
    prominences). Instead, the strongest radial edge is taken on rays from the prior centre and
    a circle is fitted with RANSAC under a radius prior. A fit is accepted when enough rays agree
    with it; otherwise (clouds, a blank frame) the prior is returned.
    """
    prior = _prior(img.shape)
    r_range = (prior.r * (1 - _RADIUS_TOL), prior.r * (1 + _RADIUS_TOL))
    for lo, hi in _SEARCH_WINDOWS:
        x, y = _limb_edges(img, prior, prior.r + lo, prior.r + hi, n_rays)
        res = _ransac_circle(x, y, r_range)
        if res is not None and res[1] >= _MIN_COVERAGE * n_rays:
            fit = res[0]
            if r_range[0] <= fit.r <= r_range[1]:
                return fit
    log.warning("limb fit failed, using the fixed GONG disk geometry")
    return prior


DISK_MARGIN = 6.0  # px between the limb ring radius and the disk mask edge


def circle_mask(shape: tuple[int, ...], d: Disk, margin: float = DISK_MARGIN) -> np.ndarray:
    """Boolean mask of the circle ``d`` shrunk by ``margin`` px."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    mask: np.ndarray = (xx - d.cx) ** 2 + (yy - d.cy) ** 2 <= (d.r - margin) ** 2
    return mask


def disk_mask(img: np.ndarray, *, margin: float = DISK_MARGIN) -> np.ndarray:
    """Boolean mask of the solar disk (always a circle), shrunk by ``margin`` px from the limb
    ring radius to drop the ring and the dark dip just inside it."""
    return circle_mask(img.shape, locate_disk(img), margin)
