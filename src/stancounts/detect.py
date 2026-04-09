"""Heuristics to detect whether data is log1p-normalized and determine the log base."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _sample_rows(X, n: int, rng: np.random.RandomState):
    """Get indices of up to *n* random rows."""
    n_rows = X.shape[0]
    if n_rows <= n:
        return np.arange(n_rows)
    return rng.choice(n_rows, n, replace=False)


def _integer_ratio_score(values: np.ndarray, tol: float = 0.01) -> float:
    """Fraction of values whose ratio to the minimum is near-integer."""
    nz = values[values > 0]
    if len(nz) < 5:
        return 0.0
    min_val = nz.min()
    ratios = nz / min_val
    deviations = np.abs(ratios - np.round(ratios))
    return float((deviations < tol).sum() / len(deviations))


def _is_integer_data(X, n_sample: int = 200, rng: np.random.RandomState | None = None) -> bool:
    """Check if the data is already integer-valued (raw counts)."""
    if rng is None:
        rng = np.random.RandomState(0)
    idx = _sample_rows(X, n_sample, rng)

    if sp.issparse(X):
        X_csr = X.tocsr()
        for i in idx:
            row = X_csr.data[X_csr.indptr[i]:X_csr.indptr[i + 1]]
            if len(row) > 0 and not np.allclose(row, np.round(row), atol=1e-6):
                return False
    else:
        for i in idx:
            row = X[i]
            nz = row[row != 0]
            if len(nz) > 0 and not np.allclose(nz, np.round(nz), atol=1e-6):
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

    # Check integer
    is_int = _is_integer_data(X, n_sample, rng)

    # Sample max value
    if sp.issparse(X):
        X_csr = X.tocsr()
        max_val = max(
            (X_csr.data[X_csr.indptr[i]:X_csr.indptr[i + 1]].max()
             for i in idx
             if X_csr.indptr[i] < X_csr.indptr[i + 1]),
            default=0.0,
        )
    else:
        max_val = float(np.max([np.max(X[i]) for i in idx]))

    # Score each base
    bases = {"e": np.expm1, "2": lambda x: np.power(2.0, x) - 1, "10": lambda x: np.power(10.0, x) - 1}
    scores = {}

    for base_name, inv_fn in bases.items():
        cell_scores = []
        for i in idx:
            if sp.issparse(X):
                row = X_csr[i].toarray().flatten()
            else:
                row = np.asarray(X[i]).flatten()

            nz = row[row > 0]
            if len(nz) < 5:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                inv = inv_fn(nz.astype(np.float64))
            # Skip if overflow produced inf/nan
            if not np.all(np.isfinite(inv)):
                continue
            score = _integer_ratio_score(inv)
            cell_scores.append(score)

        scores[base_name] = float(np.mean(cell_scores)) if cell_scores else 0.0

    best_base = max(scores, key=scores.get)
    best_score = scores[best_base]

    # Decision logic
    # 1. If data is integer AND max_val is large (>20), probably raw counts
    # 2. If best score > 0.9 and data is NOT integer, very likely log1p
    # 3. If data is integer but max_val is small (<15), could be log-normalized
    #    with integer-ish values due to float32 precision — check more carefully
    if is_int and max_val > 20:
        is_log1p = False
        detected_base = None
    elif best_score > 0.9:
        is_log1p = True
        detected_base = best_base
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
        "max_value": float(max_val),
    }


def is_log1p_normalized(X, **kwargs) -> bool:
    """Convenience wrapper: returns True if *X* appears log1p-normalized."""
    return detect_normalization(X, **kwargs)["is_log1p"]
