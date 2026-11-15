import numpy as np
import pytest

from filseg.competition.coco import IMAGE_SIZE


def blob(cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    """Axis-aligned elliptical mask on the full 2048x2048 canvas."""
    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
