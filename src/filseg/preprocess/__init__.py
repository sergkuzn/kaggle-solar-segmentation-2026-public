"""Image -> model input: solar-disk geometry (`disk`), input channels (`channels`) and the cached
`Preprocessor` (`preprocessor`)."""

from filseg.preprocess.channels import build_channels, flat_field
from filseg.preprocess.disk import GONG_DISK, Disk, circle_mask, disk_mask, locate_disk
from filseg.preprocess.preprocessor import INPUT_DTYPE, PREPROC_VERSION, Preprocessor

__all__ = [
    "GONG_DISK",
    "INPUT_DTYPE",
    "PREPROC_VERSION",
    "Disk",
    "Preprocessor",
    "build_channels",
    "circle_mask",
    "disk_mask",
    "flat_field",
    "locate_disk",
]
