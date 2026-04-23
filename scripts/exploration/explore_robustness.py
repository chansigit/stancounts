"""
Explore robustness:
1. Vectorized implementation (speed)
2. Simulated edge cases (min_count > 1, different target_sum, scran-style size factors)
3. Detection heuristics (is this data log1p-normalized?)
"""

import numpy as np
import scipy.sparse as sp
import time


# ============================================================
# 1. Vectorized reverse_log1p
# ============================================================

def reverse_log1p_fast(X):
    """Vectorized reverse log1p normalization.

    For sparse matrices, avoids Python loops by using sparse operations.
    """
    is_sparse = sp.issparse(X)

    # Step 1: expm1
    if is_sparse:
        normed = X.copy().tocsr()
        normed.data = np.expm1(normed.data).astype(np.float64)
    else:
        normed = np.expm1(np.asarray(X, dtype=np.float64))

    n_cells = normed.shape[0]

    # Step 2: min nonzero per cell (vectorized for sparse)
    if is_sparse:
        # Use the sparse structure directly
        # For each row, find the min of .data[indptr[i]:indptr[i+1]]
        # We can do this efficiently with np.minimum.reduceat
        indptr = normed.indptr
        row_lengths = np.diff(indptr)
        nonempty = row_lengths > 0

        scale_factors = np.full(n_cells, np.nan)
        if nonempty.any():
            # reduceat on non-empty rows
            starts = indptr[:-1][nonempty]
            mins = np.minimum.reduceat(normed.data, starts)
            # reduceat may include data from next row if rows are contiguous,
            # but since we only take the value at each start, it's correct
            # for the minimum within [start, start+length)
            # Actually, np.minimum.reduceat computes min over [starts[i], starts[i+1])
            # which is not exactly what we want. Let me use a different approach.

            # Safer approach: use scipy sparse operations
            # Replace all values with their row min
            # Actually the fastest safe way for sparse min per row:
            min_per_row = np.zeros(n_cells)
            data = normed.data
            for i in np.where(nonempty)[0]:
                s, e = indptr[i], indptr[i + 1]
                min_per_row[i] = data[s:e].min()
            scale_factors = min_per_row
            scale_factors[~nonempty] = np.nan
    else:
        # Dense: use masked array
        masked = np.where(normed > 0, normed, np.inf)
        scale_factors = np.min(masked, axis=1)
        scale_factors[scale_factors == np.inf] = np.nan

    # Step 3: divide and round
    if is_sparse:
        recovered = normed.copy()
        # Vectorized: create a repeat array of scale_factors aligned with data
        row_indices = np.repeat(np.arange(n_cells), row_lengths)
        sf_expanded = scale_factors[row_indices]
        valid = ~np.isnan(sf_expanded)
        recovered.data[valid] = np.round(recovered.data[valid] / sf_expanded[valid])
        recovered.data[~valid] = 0
    else:
        sf_col = scale_factors[:, np.newaxis]
        recovered = np.where(np.isnan(sf_col), 0, np.round(normed / sf_col))

    return recovered, scale_factors


# ============================================================
# 2. Test: simulate different normalization scenarios
# ============================================================

def simulate_and_test(name, counts, size_factors=None, target_sum=1e4):
    """Simulate normalization and test recovery."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    n_cells, n_genes = counts.shape
    print(f"  Shape: {n_cells} x {n_genes}")

    # Normalize
    if size_factors is None:
        lib_sizes = counts.sum(axis=1)
        size_factors = lib_sizes / target_sum

    # Normalize: count / size_factor, then log1p
    normed = counts / size_factors[:, np.newaxis]
    log1p_normed = np.log1p(normed)

    # Make sparse (typical for scRNA-seq)
    log1p_sparse = sp.csr_matrix(log1p_normed)

    # Recover
    t0 = time.time()
    recovered, sf = reverse_log1p_fast(log1p_sparse)
    t1 = time.time()

    rec_dense = recovered.toarray() if sp.issparse(recovered) else recovered

    # Evaluate
    exact = (rec_dense == counts)
    nz = counts > 0
    print(f"  Recovery time: {t1-t0:.3f}s")
    print(f"  Element-wise exact: {exact.sum()}/{exact.size} ({exact.sum()/exact.size*100:.2f}%)")
    if nz.sum() > 0:
        exact_nz = (rec_dense[nz] == counts[nz])
        print(f"  Nonzero exact: {exact_nz.sum()}/{nz.sum()} ({exact_nz.sum()/nz.sum()*100:.2f}%)")

    cell_exact = np.all(exact, axis=1)
    print(f"  Cells perfect: {cell_exact.sum()}/{n_cells} ({cell_exact.sum()/n_cells*100:.2f}%)")

    min_counts = np.array([counts[i][counts[i] > 0].min() if (counts[i] > 0).any() else 0
                           for i in range(n_cells)])
    has_1 = (min_counts == 1).sum()
    print(f"  Cells with min_count==1: {has_1}/{n_cells} ({has_1/n_cells*100:.1f}%)")

    if not np.all(exact):
        diff = np.abs(rec_dense - counts)
        print(f"  Max diff: {diff.max():.1f}, Mean diff: {diff.mean():.6f}")
        # Analyze failures
        bad_cells = ~cell_exact
        bad_idx = np.where(bad_cells)[0][:3]
        for idx in bad_idx:
            mc = min_counts[idx]
            print(f"    Cell {idx}: min_count={mc}, inferred_sf={sf[idx]:.4f}, "
                  f"true_sf={size_factors[idx]:.4f}, ratio={sf[idx]/size_factors[idx]:.4f}")


np.random.seed(42)

# --- Scenario A: Standard (every cell has count=1) ---
n, g = 1000, 5000
counts_a = np.zeros((n, g), dtype=np.float64)
# Sparse counts: ~10% nonzero, Poisson-like
for i in range(n):
    n_expressed = np.random.randint(200, 1000)
    genes = np.random.choice(g, n_expressed, replace=False)
    counts_a[i, genes] = np.random.geometric(p=0.3, size=n_expressed)
simulate_and_test("Scenario A: Standard (geometric counts)", counts_a, target_sum=1e4)

# --- Scenario B: Different target_sum (1e6 = CPM) ---
simulate_and_test("Scenario B: CPM (target_sum=1e6)", counts_a, target_sum=1e6)

# --- Scenario C: target_sum = median library size ---
lib_sizes = counts_a.sum(axis=1)
target_sum_c = np.median(lib_sizes)
print(f"\n  (Median library size = {target_sum_c:.0f})")
simulate_and_test("Scenario C: target_sum=median(lib_size)", counts_a, target_sum=target_sum_c)

# --- Scenario D: Cells where min_count > 1 (forced) ---
counts_d = counts_a.copy()
# Force some cells to have no count=1 genes
for i in range(0, 50):
    counts_d[i][counts_d[i] == 1] = 2  # replace all 1s with 2s
simulate_and_test("Scenario D: 50 cells with min_count=2 (forced)", counts_d, target_sum=1e4)

# --- Scenario E: scran-style normalization (per-cell size factors, not library-size-based) ---
# Simulate by using random size factors (not proportional to library size)
size_factors_e = np.random.lognormal(mean=0, sigma=0.5, size=n)
simulate_and_test("Scenario E: scran-style (random size factors)", counts_a,
                  size_factors=size_factors_e)

# --- Scenario F: Very deeply sequenced cells (high counts, min still 1?) ---
counts_f = np.zeros((100, 5000), dtype=np.float64)
for i in range(100):
    n_expressed = np.random.randint(3000, 4500)
    genes = np.random.choice(5000, n_expressed, replace=False)
    # High counts: geometric with small p (more counts)
    counts_f[i, genes] = np.random.geometric(p=0.05, size=n_expressed)
simulate_and_test("Scenario F: Deeply sequenced (high counts)", counts_f, target_sum=1e4)


# ============================================================
# 3. Detection heuristic: is data log1p-normalized?
# ============================================================
print("\n\n" + "="*60)
print("  Detection heuristics")
print("="*60)

# Properties of log1p-normalized data:
# 1. All values >= 0
# 2. After expm1, min nonzero per cell divides all other nonzero values (approximately integer ratios)
# 3. The distribution has a characteristic shape

import scanpy as sc
from pathlib import Path

adata = sc.read_h5ad(Path("/home/users/chensj16/s/projects/genofoundation/data/pbmc/pbmc3k.h5ad"))

# Test on actual log1p data
X = adata.X
if sp.issparse(X):
    X_csr = X.tocsr()
    sample_cell = 0
    row = np.expm1(X_csr[sample_cell].toarray().flatten())
    nz = row[row > 0]
    nz_sorted = np.sort(nz)
    min_nz = nz_sorted[0]
    ratios = nz_sorted / min_nz
    print(f"\n  Sample cell 0 (log1p data):")
    print(f"    Min nonzero (after expm1): {min_nz:.6f}")
    print(f"    First 10 ratios: {ratios[:10]}")
    print(f"    Are ratios close to integers? {np.allclose(ratios, np.round(ratios), atol=1e-3)}")
    print(f"    Max deviation from integer: {np.max(np.abs(ratios - np.round(ratios))):.6e}")

# Test on raw counts (NOT log1p) - should fail detection
raw = adata.layers["counts"]
if sp.issparse(raw):
    raw_csr = raw.tocsr()
    row_raw = raw_csr[sample_cell].toarray().flatten()
    nz_raw = row_raw[row_raw > 0]
    nz_raw_sorted = np.sort(nz_raw)
    min_nz_raw = nz_raw_sorted[0]
    ratios_raw = nz_raw_sorted / min_nz_raw
    print(f"\n  Sample cell 0 (raw counts):")
    print(f"    Min nonzero: {min_nz_raw}")
    print(f"    First 10 ratios: {ratios_raw[:10]}")
    print(f"    Are ratios close to integers? Already integers!")

# Test on random data (should fail detection)
random_data = np.random.exponential(1.0, size=(1, 5000))
random_data[random_data < 0.5] = 0
nz_rand = random_data[0][random_data[0] > 0]
nz_rand_sorted = np.sort(nz_rand)
min_nz_rand = nz_rand_sorted[0]
ratios_rand = nz_rand_sorted / min_nz_rand
print(f"\n  Random exponential data:")
print(f"    Min nonzero: {min_nz_rand:.6f}")
print(f"    First 10 ratios: {ratios_rand[:10]}")
deviation = np.abs(ratios_rand - np.round(ratios_rand))
print(f"    Are ratios close to integers? {np.allclose(ratios_rand, np.round(ratios_rand), atol=1e-3)}")
print(f"    Fraction of ratios near integer (|dev|<0.01): {(deviation < 0.01).sum()}/{len(deviation)}")
