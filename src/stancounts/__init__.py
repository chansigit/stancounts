"""stancounts: Recover raw counts from log1p-normalized single-cell expression matrices."""

from stancounts.core import reverse_log1p, reverse_log1p_anndata
from stancounts.counts import CountsUnavailable, get_counts
from stancounts.detect import detect_normalization, is_log1p_normalized

__version__ = "0.3.0"
__all__ = [
    "reverse_log1p",
    "reverse_log1p_anndata",
    "get_counts",
    "CountsUnavailable",
    "detect_normalization",
    "is_log1p_normalized",
]
