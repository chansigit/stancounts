"""Heuristics to detect whether data is log1p-normalized and determine the log base."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from stancounts.core import INV_LOG1P


def _sample_rows(X, n: int, rng: np.random.RandomState):
    """Get indices of up to *n* random rows."""
    n_rows = X.shape[0]
    if n_rows <= n:
        return np.arange(n_rows)
    return rng.choice(n_rows, n, replace=False)


def _row_nonzero(X, i, X_csr=None):
    """Return the nonzero values of row *i* as a 1-D ndarray."""
    if X_csr is not None:
        s, e = X_csr.indptr[i], X_csr.indptr[i + 1]
        return X_csr.data[s:e]
    row = np.asarray(X[i]).ravel()
    return row[row != 0]


def _integer_ratio_score(values: np.ndarray, tol: float = 0.01) -> float:
    """Fraction of values whose ratio to the minimum is near-integer."""
    nz = values[values > 0]
    if len(nz) < 5:
        return 0.0
    min_val = nz.min()
    ratios = nz / min_val
    deviations = np.abs(ratios - np.round(ratios))
    return float((deviations < tol).sum() / len(deviations))


def _is_integer_data(X, X_csr, idx) -> bool:
    """Check if the data is already integer-valued (raw counts)."""
    for i in idx:
        row = _row_nonzero(X, i, X_csr)
        if len(row) > 0 and not np.allclose(row, np.round(row), atol=1e-6):
            return False
    return True


def detect_normalization(
    X,
    *,
    n_sample: int = 200,
    seed: int = 0,
) -> dict:
    """Detect whether *X* is log1p-normalized and determine the log base.

    Parameters
    ----------
    X : array-like or sparse matrix
        Expression matrix (cells x genes).
    n_sample : int
        Number of cells to sample for the heuristic.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``is_log1p`` : bool
            Whether the data appears to be log1p-normalized.
        ``base`` : str or None
            Best-guess log base (``'e'``, ``'2'``, ``'10'``) or None.
        ``scores`` : dict
            Per-base integer-ratio scores (higher = more likely).
        ``is_integer`` : bool
            Whether the raw data is already integer-valued.
        ``max_value`` : float
            Maximum value in the sampled data.
    """
    rng = np.random.RandomState(seed)
    idx = _sample_rows(X, n_sample, rng)

    X_csr = X.tocsr() if sp.issparse(X) else None

    is_int = _is_integer_data(X, X_csr, idx)

    # Sample max value and cache per-row nonzero vectors (reused across bases)
    rows = [_row_nonzero(X, i, X_csr) for i in idx]
    max_val = float(max((r.max() for r in rows if r.size), default=0.0))

    scores = {}
    for base_name, inv_fn in INV_LOG1P.items():
        cell_scores = []
        for nz in rows:
            if len(nz) < 5:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                inv = inv_fn(nz.astype(np.float64))
            if not np.all(np.isfinite(inv)):
                continue
            cell_scores.append(_integer_ratio_score(inv))
        scores[base_name] = float(np.mean(cell_scores)) if cell_scores else 0.0

    best_base, best_score = max(scores.items(), key=lambda kv: kv[1])

    # Decision:
    # - Integer data with large max → raw counts
    # - Otherwise, a best-base score above 0.7 indicates log1p
    if is_int and max_val > 20:
        is_log1p = False
        detected_base = None
    elif best_score > 0.7:
        is_log1p = True
        detected_base = best_base
    else:
        is_log1p = False
        detected_base = None

    return {
        "is_log1p": is_log1p,
        "base": detected_base,
        "scores": scores,
        "is_integer": is_int,
        "max_value": max_val,
    }


def is_log1p_normalized(X, **kwargs) -> bool:
    """Convenience wrapper: returns True if *X* appears log1p-normalized."""
    return detect_normalization(X, **kwargs)["is_log1p"]
