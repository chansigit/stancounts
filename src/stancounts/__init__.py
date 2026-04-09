"""stancounts: Recover raw counts from log1p-normalized single-cell expression matrices."""

from stancounts.core import reverse_log1p, reverse_log1p_anndata
from stancounts.detect import detect_normalization, is_log1p_normalized

__version__ = "0.1.0"
__all__ = [
    "reverse_log1p",
    "reverse_log1p_anndata",
    "detect_normalization",
    "is_log1p_normalized",
]
