"""
Explore reversing log1p normalization to recover raw counts.

Math:
  Forward: y = log1p(count / library_size * target_sum)
  Reverse:
    1. expm1(y) -> count / library_size * target_sum  (= count * scale_factor)
    2. min nonzero per cell -> corresponds to count=1 -> gives scale_factor
    3. round(expm1(y) / scale_factor) -> recovered counts
"""

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

DATA_DIR = Path("/home/users/chensj16/s/projects/genofoundation/data/pbmc")

# --- Load PBMC3k ---
print("Loading pbmc3k...")
adata = sc.read_h5ad(DATA_DIR / "pbmc3k.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  X dtype: {adata.X.dtype}, type: {type(adata.X)}")
print(f"  layers: {list(adata.layers.keys())}")

# Ground truth counts
counts_true = adata.layers["counts"]
log1p_norm = adata.X  # log1p normalized (target_sum=1e4)

print(f"\n  counts_true type: {type(counts_true)}, dtype: {counts_true.dtype}")
print(f"  log1p_norm type: {type(log1p_norm)}, dtype: {log1p_norm.dtype}")

# --- Step 1: expm1 to undo log1p ---
print("\n=== Step 1: expm1 ===")
if sp.issparse(log1p_norm):
    # For sparse matrix, operate on .data directly
    normed = log1p_norm.copy()
    normed.data = np.expm1(normed.data)
else:
    normed = np.expm1(log1p_norm)

print(f"  normed type: {type(normed)}, shape: {normed.shape}")

# --- Step 2: Find min nonzero per cell ---
print("\n=== Step 2: Find min nonzero per cell (= scale_factor for count=1) ===")

n_cells = normed.shape[0]
min_nonzero = np.zeros(n_cells)

if sp.issparse(normed):
    normed_csr = normed.tocsr()
    for i in range(n_cells):
        row = normed_csr[i].data
        if len(row) > 0:
            min_nonzero[i] = row.min()
        else:
            min_nonzero[i] = np.nan
else:
    for i in range(n_cells):
        row = normed[i]
        nonzero_vals = row[row > 0]
        if len(nonzero_vals) > 0:
            min_nonzero[i] = nonzero_vals.min()
        else:
            min_nonzero[i] = np.nan

print(f"  min_nonzero stats: min={np.nanmin(min_nonzero):.6f}, "
      f"max={np.nanmax(min_nonzero):.6f}, "
      f"median={np.nanmedian(min_nonzero):.6f}")
print(f"  NaN cells: {np.isnan(min_nonzero).sum()}")

# --- Verify: min_nonzero should equal target_sum / library_size ---
print("\n=== Verification: compare inferred scale_factor with true values ===")
if sp.issparse(counts_true):
    lib_sizes_true = np.array(counts_true.sum(axis=1)).flatten()
else:
    lib_sizes_true = counts_true.sum(axis=1)

target_sum = 1e4
scale_factor_true = target_sum / lib_sizes_true
scale_factor_inferred = min_nonzero

print(f"  True scale_factor: min={scale_factor_true.min():.6f}, max={scale_factor_true.max():.6f}")
print(f"  Inferred scale_factor: min={np.nanmin(scale_factor_inferred):.6f}, max={np.nanmax(scale_factor_inferred):.6f}")

ratio = scale_factor_inferred / scale_factor_true
print(f"  Ratio (inferred/true): min={np.nanmin(ratio):.6f}, max={np.nanmax(ratio):.6f}, "
      f"median={np.nanmedian(ratio):.6f}")
print(f"  Cells where ratio == 1.0: {np.sum(np.isclose(ratio, 1.0))}/{n_cells} "
      f"({np.sum(np.isclose(ratio, 1.0))/n_cells*100:.1f}%)")

# --- Step 3: Recover counts ---
print("\n=== Step 3: Recover counts ===")
if sp.issparse(normed):
    normed_csr = normed.tocsr()
    recovered = normed_csr.copy()
    for i in range(n_cells):
        start, end = recovered.indptr[i], recovered.indptr[i + 1]
        if end > start and not np.isnan(min_nonzero[i]):
            recovered.data[start:end] = np.round(
                recovered.data[start:end] / min_nonzero[i]
            )
    recovered = recovered.astype(np.float32)
else:
    recovered = np.zeros_like(normed)
    for i in range(n_cells):
        if not np.isnan(min_nonzero[i]):
            recovered[i] = np.round(normed[i] / min_nonzero[i])

# --- Step 4: Compare with ground truth ---
print("\n=== Step 4: Compare recovered vs. true counts ===")
if sp.issparse(counts_true):
    counts_dense = counts_true.toarray()
else:
    counts_dense = counts_true

if sp.issparse(recovered):
    recovered_dense = recovered.toarray()
else:
    recovered_dense = recovered

# Overall metrics
exact_match = (recovered_dense == counts_dense)
print(f"  Element-wise exact match: {exact_match.sum()}/{exact_match.size} "
      f"({exact_match.sum()/exact_match.size*100:.4f}%)")

# For nonzero elements only
nonzero_mask = counts_dense > 0
if nonzero_mask.sum() > 0:
    exact_nonzero = (recovered_dense[nonzero_mask] == counts_dense[nonzero_mask])
    print(f"  Nonzero exact match: {exact_nonzero.sum()}/{nonzero_mask.sum()} "
          f"({exact_nonzero.sum()/nonzero_mask.sum()*100:.4f}%)")

# Per-cell accuracy
cell_exact = np.all(exact_match, axis=1)
print(f"  Cells perfectly recovered: {cell_exact.sum()}/{n_cells} "
      f"({cell_exact.sum()/n_cells*100:.1f}%)")

# Where they differ
diff = np.abs(recovered_dense - counts_dense)
if diff.max() > 0:
    print(f"\n  Max absolute difference: {diff.max():.1f}")
    print(f"  Mean absolute difference (nonzero elements): {diff[nonzero_mask].mean():.6f}")

    # Distribution of errors
    errors = diff[diff > 0]
    print(f"  Error distribution:")
    for threshold in [0.5, 1, 2, 5, 10]:
        n = (errors > threshold).sum()
        print(f"    |diff| > {threshold}: {n} ({n/errors.size*100:.2f}% of errors)")

# Check: are the errors from cells where min_nonzero != true scale_factor?
bad_cells = ~np.isclose(ratio, 1.0)
print(f"\n  Cells with wrong scale_factor: {bad_cells.sum()}")
if bad_cells.sum() > 0:
    errors_bad = diff[bad_cells].sum()
    errors_total = diff.sum()
    print(f"  Errors from bad-scale cells: {errors_bad:.0f}/{errors_total:.0f} "
          f"({errors_bad/errors_total*100:.1f}%)")

# --- Examine failure cases ---
print("\n=== Failure analysis ===")
bad_idx = np.where(bad_cells)[0][:5]
for idx in bad_idx:
    true_counts = counts_dense[idx]
    min_true_nz = true_counts[true_counts > 0].min()
    print(f"  Cell {idx}: min_true_count={min_true_nz}, "
          f"lib_size={lib_sizes_true[idx]:.0f}, "
          f"true_scale={scale_factor_true[idx]:.6f}, "
          f"inferred_scale={scale_factor_inferred[idx]:.6f}, "
          f"ratio={ratio[idx]:.4f}")
