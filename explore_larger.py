"""
Test reverse log1p on purified_pbmc and cite_seq_pbmc.
Also explore edge cases: cells where min count > 1.
"""

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path
import time

DATA_DIR = Path("/home/users/chensj16/s/projects/genofoundation/data/pbmc")


def reverse_log1p_counts(X):
    """Reverse log1p normalization to recover counts.

    Args:
        X: log1p-normalized expression matrix (cells x genes), sparse or dense

    Returns:
        recovered: recovered count matrix
        scale_factors: inferred scale factor per cell
    """
    # Step 1: expm1
    if sp.issparse(X):
        normed = X.copy().tocsr()
        normed.data = np.expm1(normed.data)
    else:
        normed = np.expm1(X)

    n_cells = normed.shape[0]
    scale_factors = np.zeros(n_cells)

    # Step 2: min nonzero per cell
    if sp.issparse(normed):
        for i in range(n_cells):
            row_data = normed.data[normed.indptr[i]:normed.indptr[i + 1]]
            if len(row_data) > 0:
                scale_factors[i] = row_data.min()
            else:
                scale_factors[i] = np.nan
    else:
        for i in range(n_cells):
            nz = normed[i][normed[i] > 0]
            scale_factors[i] = nz.min() if len(nz) > 0 else np.nan

    # Step 3: divide and round
    if sp.issparse(normed):
        recovered = normed.copy()
        for i in range(n_cells):
            start, end = recovered.indptr[i], recovered.indptr[i + 1]
            if end > start and not np.isnan(scale_factors[i]):
                recovered.data[start:end] = np.round(
                    recovered.data[start:end] / scale_factors[i]
                )
    else:
        recovered = np.zeros_like(normed)
        for i in range(n_cells):
            if not np.isnan(scale_factors[i]):
                recovered[i] = np.round(normed[i] / scale_factors[i])

    return recovered, scale_factors


def evaluate(recovered, counts_true, name=""):
    """Evaluate recovery accuracy."""
    if sp.issparse(counts_true):
        counts_dense = counts_true.toarray()
    else:
        counts_dense = np.asarray(counts_true)

    if sp.issparse(recovered):
        rec_dense = recovered.toarray()
    else:
        rec_dense = np.asarray(recovered)

    n_cells = counts_dense.shape[0]
    exact = (rec_dense == counts_dense)
    nonzero_mask = counts_dense > 0

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape: {counts_dense.shape}")
    print(f"  Element-wise exact: {exact.sum()}/{exact.size} "
          f"({exact.sum()/exact.size*100:.4f}%)")
    if nonzero_mask.sum() > 0:
        exact_nz = (rec_dense[nonzero_mask] == counts_dense[nonzero_mask])
        print(f"  Nonzero exact: {exact_nz.sum()}/{nonzero_mask.sum()} "
              f"({exact_nz.sum()/nonzero_mask.sum()*100:.4f}%)")
    cell_exact = np.all(exact, axis=1)
    print(f"  Cells perfect: {cell_exact.sum()}/{n_cells} "
          f"({cell_exact.sum()/n_cells*100:.2f}%)")

    diff = np.abs(rec_dense - counts_dense)
    if diff.max() > 0:
        print(f"  Max absolute diff: {diff.max():.1f}")
        print(f"  Mean absolute diff (all): {diff.mean():.6f}")
        print(f"  Mean absolute diff (nonzero): {diff[nonzero_mask].mean():.6f}")


def check_min_count_distribution(counts_true, name=""):
    """Check distribution of minimum nonzero count per cell."""
    if sp.issparse(counts_true):
        counts_csr = counts_true.tocsr()
        n_cells = counts_csr.shape[0]
        min_counts = np.zeros(n_cells)
        for i in range(n_cells):
            row_data = counts_csr.data[counts_csr.indptr[i]:counts_csr.indptr[i + 1]]
            if len(row_data) > 0:
                min_counts[i] = row_data.min()
            else:
                min_counts[i] = np.nan
    else:
        min_counts = np.array([
            counts_true[i][counts_true[i] > 0].min() if (counts_true[i] > 0).any() else np.nan
            for i in range(counts_true.shape[0])
        ])

    print(f"\n  {name} - Min nonzero count per cell:")
    valid = min_counts[~np.isnan(min_counts)]
    for v in sorted(np.unique(valid)):
        n = (valid == v).sum()
        print(f"    count={int(v)}: {n} cells ({n/len(valid)*100:.2f}%)")
        if v >= 5:
            print(f"    (skipping higher values...)")
            break
    return min_counts


# === PBMC3k (small, quick) ===
print("Loading pbmc3k...")
adata = sc.read_h5ad(DATA_DIR / "pbmc3k.h5ad")
check_min_count_distribution(adata.layers["counts"], "PBMC3k")

t0 = time.time()
recovered, sf = reverse_log1p_counts(adata.X)
t1 = time.time()
evaluate(recovered, adata.layers["counts"], f"PBMC3k ({t1-t0:.1f}s)")
del adata

# === Purified PBMC (medium) ===
print("\n\nLoading purified_pbmc...")
adata = sc.read_h5ad(DATA_DIR / "purified_pbmc.h5ad")
check_min_count_distribution(adata.layers["counts"], "Purified PBMC")

t0 = time.time()
recovered, sf = reverse_log1p_counts(adata.X)
t1 = time.time()
evaluate(recovered, adata.layers["counts"], f"Purified PBMC ({t1-t0:.1f}s)")
del adata

# === CITE-seq PBMC (large) ===
print("\n\nLoading cite_seq_pbmc...")
adata = sc.read_h5ad(DATA_DIR / "cite_seq_pbmc.h5ad")
check_min_count_distribution(adata.layers["counts"], "CITE-seq PBMC")

t0 = time.time()
recovered, sf = reverse_log1p_counts(adata.X)
t1 = time.time()
evaluate(recovered, adata.layers["counts"], f"CITE-seq PBMC ({t1-t0:.1f}s)")
del adata
