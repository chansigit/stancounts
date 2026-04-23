"""
Explore:
1. log2(x+1) vs ln(x+1) - detect and handle different log bases
2. Robust scale factor inference (handle min_count > 1)
3. Full detection pipeline: is this data log1p-normalized? what base?
"""

import numpy as np
import scipy.sparse as sp
import time


def make_sparse_counts(n_cells=500, n_genes=3000, seed=42):
    """Generate realistic sparse count matrix."""
    rng = np.random.RandomState(seed)
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    for i in range(n_cells):
        n_expr = rng.randint(200, 1000)
        genes = rng.choice(n_genes, n_expr, replace=False)
        counts[i, genes] = rng.geometric(p=0.3, size=n_expr)
    return counts


def normalize_log1p(counts, target_sum=1e4, base="e"):
    """Normalize counts: scale by size factor, then log(1+x)."""
    lib_sizes = counts.sum(axis=1, keepdims=True)
    normed = counts / lib_sizes * target_sum
    if base == "e":
        return np.log1p(normed)
    elif base == "2":
        return np.log2(1 + normed)
    elif base == "10":
        return np.log10(1 + normed)
    else:
        raise ValueError(f"Unknown base: {base}")


def inv_log1p(X, base="e"):
    """Inverse of log(1+x) for different bases."""
    if base == "e":
        return np.expm1(X)
    elif base == "2":
        return np.power(2, X) - 1
    elif base == "10":
        return np.power(10, X) - 1
    else:
        raise ValueError


def detect_log_base(X, n_sample_cells=100, n_check=50):
    """Detect whether data is log-transformed and what base was used.

    Strategy: for each candidate base, apply inverse, check if
    values/min form near-integer ratios.
    """
    if sp.issparse(X):
        X_csr = X.tocsr()
    else:
        X_csr = X

    n_cells = X.shape[0]
    sample_idx = np.random.choice(n_cells, min(n_sample_cells, n_cells), replace=False)

    results = {}
    for base in ["e", "2", "10"]:
        int_fracs = []
        for idx in sample_idx:
            if sp.issparse(X_csr):
                row = X_csr[idx].toarray().flatten()
            else:
                row = X_csr[idx].copy()

            nz = row[row > 0]
            if len(nz) < 10:
                continue

            inv = inv_log1p(nz, base)
            min_inv = inv.min()
            ratios = inv / min_inv

            # Check how many ratios are close to integers
            deviations = np.abs(ratios - np.round(ratios))
            frac_int = (deviations < 0.01).sum() / len(deviations)
            int_fracs.append(frac_int)

        mean_frac = np.mean(int_fracs) if int_fracs else 0
        results[base] = mean_frac

    return results


def recover_counts_robust(X, base="e"):
    """Recover counts with robust scale factor estimation.

    Handles the rare case where min_count > 1 by checking
    whether ratios suggest we need to halve the scale factor.
    """
    is_sparse = sp.issparse(X)

    # Step 1: inverse log
    if is_sparse:
        normed = X.copy().tocsr()
        normed.data = inv_log1p(normed.data, base).astype(np.float64)
    else:
        normed = inv_log1p(np.asarray(X, dtype=np.float64), base)

    n_cells = normed.shape[0]

    # Step 2: per-cell scale factor inference
    scale_factors = np.zeros(n_cells)
    corrections = np.ones(n_cells, dtype=int)  # multiplier for min_count > 1 correction

    if is_sparse:
        normed_csr = normed.tocsr() if not isinstance(normed, sp.csr_matrix) else normed
        for i in range(n_cells):
            s, e = normed_csr.indptr[i], normed_csr.indptr[i + 1]
            if s == e:
                scale_factors[i] = np.nan
                continue

            row_data = normed_csr.data[s:e]
            min_val = row_data.min()
            scale_factors[i] = min_val

            # Robustness check: do ratios suggest min_count > 1?
            # Sample some values to check
            if len(row_data) > 10:
                sample = row_data[:min(50, len(row_data))]
                ratios = sample / min_val
                deviations = np.abs(ratios - np.round(ratios))

                # If many half-integer ratios, scale factor is 2x too large
                half_int = np.abs(deviations - 0.5) < 0.05
                if half_int.sum() / len(half_int) > 0.15:
                    scale_factors[i] = min_val / 2
                    corrections[i] = 2

                # Could extend to 3x, but extremely rare
    else:
        for i in range(n_cells):
            nz = normed[i][normed[i] > 0]
            if len(nz) == 0:
                scale_factors[i] = np.nan
                continue
            min_val = nz.min()
            scale_factors[i] = min_val

            if len(nz) > 10:
                ratios = nz[:min(50, len(nz))] / min_val
                deviations = np.abs(ratios - np.round(ratios))
                half_int = np.abs(deviations - 0.5) < 0.05
                if half_int.sum() / len(half_int) > 0.15:
                    scale_factors[i] = min_val / 2
                    corrections[i] = 2

    # Step 3: recover
    if is_sparse:
        recovered = normed.copy()
        indptr = recovered.indptr
        row_lengths = np.diff(indptr)
        row_indices = np.repeat(np.arange(n_cells), row_lengths)
        sf_expanded = scale_factors[row_indices]
        valid = ~np.isnan(sf_expanded) & (sf_expanded > 0)
        recovered.data[valid] = np.round(recovered.data[valid] / sf_expanded[valid])
        recovered.data[~valid] = 0
    else:
        sf_col = scale_factors[:, np.newaxis]
        valid = ~np.isnan(sf_col) & (sf_col > 0)
        recovered = np.where(valid, np.round(normed / sf_col), 0)

    return recovered, scale_factors, corrections


# ============================================================
# Test 1: Different log bases
# ============================================================
counts = make_sparse_counts()

for base in ["e", "2", "10"]:
    log_data = normalize_log1p(counts, target_sum=1e4, base=base)
    log_sparse = sp.csr_matrix(log_data)

    # Detect
    det = detect_log_base(log_sparse)
    best_base = max(det, key=det.get)

    # Recover
    rec, sf, corr = recover_counts_robust(log_sparse, base=base)
    rec_dense = rec.toarray() if sp.issparse(rec) else rec

    exact = (rec_dense == counts).sum() / counts.size * 100
    print(f"Base={base:>2s}: detection={det}, best={best_base}, recovery={exact:.2f}%")


# ============================================================
# Test 2: Robust recovery for min_count > 1
# ============================================================
print("\n=== Robust recovery for min_count > 1 ===")
counts_bad = counts.copy()
# Force first 50 cells to have no count=1
for i in range(50):
    counts_bad[i][counts_bad[i] == 1] = 2

log_bad = normalize_log1p(counts_bad, target_sum=1e4, base="e")
log_bad_sparse = sp.csr_matrix(log_bad)

# Without robustness
from explore_robustness import reverse_log1p_fast
rec_naive, sf_naive = reverse_log1p_fast(log_bad_sparse)
rec_naive_dense = rec_naive.toarray() if sp.issparse(rec_naive) else rec_naive
exact_naive = np.all(rec_naive_dense == counts_bad, axis=1)

# With robustness
rec_robust, sf_robust, corrections = recover_counts_robust(log_bad_sparse, base="e")
rec_robust_dense = rec_robust.toarray() if sp.issparse(rec_robust) else rec_robust
exact_robust = np.all(rec_robust_dense == counts_bad, axis=1)

print(f"  Naive:  {exact_naive.sum()}/500 cells perfect ({exact_naive.sum()/500*100:.1f}%)")
print(f"  Robust: {exact_robust.sum()}/500 cells perfect ({exact_robust.sum()/500*100:.1f}%)")
print(f"  Cells corrected (min_count>1 detected): {(corrections > 1).sum()}")

# Check which of the 50 bad cells were fixed
bad_mask = np.zeros(500, dtype=bool)
bad_mask[:50] = True
fixed_by_robust = exact_robust[bad_mask].sum()
print(f"  Bad cells fixed by robust: {fixed_by_robust}/50")


# ============================================================
# Test 3: What about data that is NOT log1p-normalized?
# ============================================================
print("\n=== Detection on non-log1p data ===")

# Raw counts
raw_sparse = sp.csr_matrix(counts)
det_raw = detect_log_base(raw_sparse)
print(f"  Raw counts: {det_raw}")

# Just scaled (no log)
lib_sizes = counts.sum(axis=1, keepdims=True)
scaled = counts / lib_sizes * 1e4
scaled_sparse = sp.csr_matrix(scaled)
det_scaled = detect_log_base(scaled_sparse)
print(f"  Scaled (no log): {det_scaled}")

# Random continuous
rng = np.random.RandomState(123)
random_data = sp.random(500, 3000, density=0.2, format="csr", random_state=rng)
det_random = detect_log_base(random_data)
print(f"  Random continuous: {det_random}")

# log1p-normalized (should be high)
log_data = normalize_log1p(counts, target_sum=1e4, base="e")
log_sparse = sp.csr_matrix(log_data)
det_log = detect_log_base(log_sparse)
print(f"  Log1p-normalized: {det_log}")


# ============================================================
# Test 4: Various target_sum values
# ============================================================
print("\n=== Different target_sum values ===")
for ts in [100, 1000, 1e4, 1e5, 1e6, None]:
    if ts is None:
        # Use median library size
        lib_sizes = counts.sum(axis=1)
        ts_actual = np.median(lib_sizes)
        label = f"median({ts_actual:.0f})"
    else:
        ts_actual = ts
        label = f"{ts_actual:.0f}"

    log_data = normalize_log1p(counts, target_sum=ts_actual, base="e")
    log_sparse = sp.csr_matrix(log_data)
    rec, sf, corr = recover_counts_robust(log_sparse, base="e")
    rec_dense = rec.toarray() if sp.issparse(rec) else rec

    exact = (rec_dense == counts).sum() / counts.size * 100
    cell_exact = np.all(rec_dense == counts, axis=1).sum()
    print(f"  target_sum={label:>12s}: {exact:.4f}% elements, {cell_exact}/500 cells perfect")
