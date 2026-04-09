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


def _inv_log1p(x: np.ndarray, base: str = "e") -> np.ndarray:
    """Compute inverse of log_base(1 + x)."""
    if base == "e":
        return np.expm1(x)
    elif base == "2":
        return np.power(2.0, x) - 1.0
    elif base == "10":
        return np.power(10.0, x) - 1.0
    else:
        raise ValueError(f"Unsupported log base: {base!r}. Use 'e', '2', or '10'.")


def _min_nonzero_per_row_sparse(csr: sp.csr_matrix) -> np.ndarray:
    """Find min nonzero value per row of a CSR matrix. NaN for empty rows."""
    n = csr.shape[0]
    mins = np.full(n, np.nan)
    data = csr.data
    indptr = csr.indptr
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if s < e:
            mins[i] = data[s:e].min()
    return mins


def _detect_half_integer_rows(csr: sp.csr_matrix, scale_factors: np.ndarray,
                              threshold: float = 0.15,
                              max_check: int = 100) -> np.ndarray:
    """Detect rows where scale_factor should be halved (min_count was 2, not 1).

    Checks whether many values/scale_factor fall near half-integers (1.5, 2.5, ...),
    which indicates the inferred scale_factor is 2x the true value.
    """
    n = csr.shape[0]
    corrections = np.ones(n, dtype=np.int32)
    data = csr.data
    indptr = csr.indptr

    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if e - s < 10 or np.isnan(scale_factors[i]):
            continue
        chunk = data[s:min(s + max_check, e)]
        ratios = chunk / scale_factors[i]
        deviations = np.abs(ratios - np.round(ratios))
        half_int_frac = (np.abs(deviations - 0.5) < 0.05).sum() / len(deviations)
        if half_int_frac > threshold:
            corrections[i] = 2
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
            Recovered integer count matrix.
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
        normed = X.copy().tocsr()
        normed.data = _inv_log1p(normed.data.astype(np.float64), base)
    else:
        normed = _inv_log1p(np.asarray(X, dtype=np.float64), base)

    n_cells = normed.shape[0]

    # --- Step 2: min nonzero per cell = scale factor ---
    if is_sparse:
        normed_csr = normed if isinstance(normed, sp.csr_matrix) else normed.tocsr()
        scale_factors = _min_nonzero_per_row_sparse(normed_csr)
    else:
        masked = np.where(normed > 0, normed, np.inf)
        scale_factors = np.min(masked, axis=1)
        scale_factors[scale_factors == np.inf] = np.nan

    # --- Step 2b: robust correction for min_count > 1 ---
    if robust and is_sparse:
        corrections = _detect_half_integer_rows(normed_csr, scale_factors)
        scale_factors = scale_factors / corrections
    elif robust and not is_sparse:
        # Dense path — convert to sparse temporarily for the check
        normed_csr_tmp = sp.csr_matrix(normed)
        corrections = _detect_half_integer_rows(normed_csr_tmp, scale_factors)
        scale_factors = scale_factors / corrections
    else:
        corrections = np.ones(n_cells, dtype=np.int32)

    # --- Step 3: divide and round ---
    if is_sparse:
        recovered = normed_csr.copy()
        indptr = recovered.indptr
        row_lengths = np.diff(indptr)
        row_indices = np.repeat(np.arange(n_cells), row_lengths)
        sf_expanded = scale_factors[row_indices]
        valid = ~np.isnan(sf_expanded) & (sf_expanded > 0)
        recovered.data[valid] = np.round(recovered.data[valid] / sf_expanded[valid])
        recovered.data[~valid] = 0
        # Ensure integer dtype
        recovered.data = recovered.data.astype(np.float32)
    else:
        sf_col = scale_factors[:, np.newaxis]
        valid = ~np.isnan(sf_col) & (sf_col > 0)
        recovered = np.where(valid, np.round(normed / sf_col), 0).astype(np.float32)

    library_sizes = np.asarray(recovered.sum(axis=1)).flatten() if is_sparse else recovered.sum(axis=1)

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
