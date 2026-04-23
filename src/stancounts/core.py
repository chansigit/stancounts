"""Core algorithm for reversing log1p normalization to recover raw counts.

Math
----
Standard single-cell normalization:

    y = log1p(count / library_size * target_sum)

Reverse:

    1. expm1(y)  →  count * (target_sum / library_size) = count * scale_factor
    2. Per-cell min nonzero after expm1 ≈ scale_factor  (assumes min count = 1)
    3. round(expm1(y) / scale_factor)  →  recovered counts

The algorithm is agnostic to target_sum and works with any per-cell size
factor normalization (scanpy, scran, Seurat, etc.).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp

# Inverse of log_base(1+x), shared with detect.py
INV_LOG1P = {
    "e": np.expm1,
    "2": lambda x: np.power(2.0, x) - 1.0,
    "10": lambda x: np.power(10.0, x) - 1.0,
}


def _inv_log1p(x: np.ndarray, base: str = "e") -> np.ndarray:
    try:
        return INV_LOG1P[base](x)
    except KeyError:
        raise ValueError(f"Unsupported log base: {base!r}. Use 'e', '2', or '10'.")


def _min_nonzero_per_row_sparse(csr: sp.csr_matrix) -> np.ndarray:
    """Vectorized min-nonzero per row of CSR. NaN for empty rows."""
    n = csr.shape[0]
    mins = np.full(n, np.nan)
    data, indptr = csr.data, csr.indptr
    if data.size == 0:
        return mins
    nonempty = np.diff(indptr) > 0
    # reduceat requires all indices < len(data); substitute 0 for empty rows then mask out
    safe_idx = np.where(nonempty, indptr[:-1], 0)
    reduced = np.minimum.reduceat(data, safe_idx)
    mins[nonempty] = reduced[nonempty]
    return mins


def _detect_half_integer_rows(
    csr: sp.csr_matrix,
    scale_factors: np.ndarray,
    threshold: float = 0.15,
    min_nnz: int = 10,
) -> np.ndarray:
    """Detect rows where scale_factor should be halved (min_count was 2, not 1).

    Vectorized: for every nonzero entry compute ratio / scale_factor and check
    how close it is to a half-integer (…, 0.5, 1.5, …). Rows with a large
    fraction of such entries have their correction set to 2.
    """
    n = csr.shape[0]
    data, indptr = csr.data, csr.indptr
    corrections = np.ones(n, dtype=np.int32)
    if data.size == 0:
        return corrections

    row_lengths = np.diff(indptr)
    row_ids = np.repeat(np.arange(n), row_lengths)
    sf = scale_factors[row_ids]
    valid = np.isfinite(sf) & (sf > 0)

    ratios = np.divide(data, sf, out=np.zeros_like(data, dtype=np.float64), where=valid)
    devs = np.abs(ratios - np.round(ratios))
    is_half = (np.abs(devs - 0.5) < 0.05) & valid

    totals = np.bincount(row_ids, weights=valid.astype(np.float64), minlength=n)
    halves = np.bincount(row_ids, weights=is_half.astype(np.float64), minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(totals >= min_nnz, halves / np.maximum(totals, 1), 0.0)
    corrections[frac > threshold] = 2
    return corrections


def reverse_log1p(
    X,
    *,
    base: Literal["e", "2", "10"] = "e",
    robust: bool = True,
) -> dict:
    """Reverse log1p normalization to recover raw counts.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_cells, n_genes)
        Log1p-normalized expression matrix. Accepts dense arrays,
        scipy sparse matrices, or anything convertible via ``np.asarray``.
    base : ``'e'``, ``'2'``, or ``'10'``
        Logarithm base used in the normalization.
    robust : bool
        If True, detect and correct for cells whose minimum nonzero count
        is >1 (rare but possible in very deeply sequenced cells).

    Returns
    -------
    dict with keys:
        ``counts`` : sparse CSR matrix or dense array, same shape as X
            Recovered integer count matrix (int32).
        ``scale_factors`` : ndarray, shape (n_cells,)
            Inferred normalization scale factor per cell
            (= target_sum / library_size for standard normalization).
        ``library_sizes`` : ndarray, shape (n_cells,)
            Recovered library size per cell (= row sums of counts).
        ``corrections`` : ndarray of int, shape (n_cells,)
            Per-cell correction multiplier (1 = no correction, 2 = halved scale factor).
    """
    is_sparse = sp.issparse(X)

    # --- Step 1: inverse log ---
    if is_sparse:
        normed = X.tocsr(copy=True)
        normed.data = _inv_log1p(normed.data.astype(np.float64), base)
    else:
        normed = _inv_log1p(np.asarray(X, dtype=np.float64), base)

    n_cells = normed.shape[0]

    # --- Step 2: min nonzero per cell = scale factor ---
    if is_sparse:
        scale_factors = _min_nonzero_per_row_sparse(normed)
    else:
        masked = np.where(normed > 0, normed, np.inf)
        scale_factors = np.min(masked, axis=1)
        scale_factors[np.isinf(scale_factors)] = np.nan

    # --- Step 2b: robust correction for min_count > 1 ---
    if robust:
        csr_for_check = normed if is_sparse else sp.csr_matrix(normed)
        corrections = _detect_half_integer_rows(csr_for_check, scale_factors)
        scale_factors = scale_factors / corrections
    else:
        corrections = np.ones(n_cells, dtype=np.int32)

    # --- Step 3: divide and round ---
    if is_sparse:
        indptr = normed.indptr
        row_ids = np.repeat(np.arange(n_cells), np.diff(indptr))
        sf_expanded = scale_factors[row_ids]
        valid = np.isfinite(sf_expanded) & (sf_expanded > 0)
        new_data = np.zeros_like(normed.data)
        new_data[valid] = np.round(normed.data[valid] / sf_expanded[valid])
        recovered = sp.csr_matrix((new_data.astype(np.int32), normed.indices.copy(), indptr.copy()),
                                  shape=normed.shape)
        recovered.eliminate_zeros()
    else:
        sf_col = scale_factors[:, np.newaxis]
        valid = np.isfinite(sf_col) & (sf_col > 0)
        recovered = np.where(valid, np.round(normed / np.where(valid, sf_col, 1.0)), 0).astype(np.int32)

    library_sizes = np.asarray(recovered.sum(axis=1)).ravel()

    return {
        "counts": recovered,
        "scale_factors": scale_factors,
        "library_sizes": library_sizes,
        "corrections": corrections,
    }


def reverse_log1p_anndata(
    adata,
    *,
    source: str = "X",
    target_layer: str = "counts_recovered",
    base: Literal["e", "2", "10"] = "e",
    robust: bool = True,
):
    """Reverse log1p normalization and store results in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in place.
    source : str
        Which slot to read. ``'X'`` reads ``adata.X``; any other string
        reads ``adata.layers[source]``.
    target_layer : str
        Layer name to store the recovered counts.
    base, robust
        Passed to :func:`reverse_log1p`.

    Returns
    -------
    dict
        Same as :func:`reverse_log1p`, plus the AnnData is modified in place
        with recovered counts in ``adata.layers[target_layer]`` and
        scale factors in ``adata.obs['stancounts_scale_factor']``.
    """
    X = adata.X if source == "X" else adata.layers[source]
    result = reverse_log1p(X, base=base, robust=robust)

    adata.layers[target_layer] = result["counts"]
    adata.obs["stancounts_scale_factor"] = result["scale_factors"]
    adata.obs["stancounts_library_size"] = result["library_sizes"]

    return result
