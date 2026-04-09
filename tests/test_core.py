"""Tests for stancounts core algorithm.

Tests against real PBMC datasets (ground truth) and simulated edge cases.
"""

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

import stancounts
from stancounts.core import reverse_log1p

PBMC_DIR = Path("/home/users/chensj16/s/projects/genofoundation/data/pbmc")
CHONDRO_DIR = Path("/home/users/chensj16/s/data/chondro-atlas/h5ad")


# ---- Helpers ----

def _make_counts(n_cells=200, n_genes=1000, seed=42):
    """Generate a realistic sparse count matrix."""
    rng = np.random.RandomState(seed)
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    for i in range(n_cells):
        n_expr = rng.randint(100, min(500, n_genes))
        genes = rng.choice(n_genes, n_expr, replace=False)
        counts[i, genes] = rng.geometric(p=0.3, size=n_expr)
    return counts


def _normalize_log1p(counts, target_sum=1e4, base="e"):
    lib = counts.sum(axis=1, keepdims=True)
    normed = counts / lib * target_sum
    if base == "e":
        return np.log1p(normed)
    elif base == "2":
        return np.log2(1 + normed)
    elif base == "10":
        return np.log10(1 + normed)


def _assert_perfect_recovery(recovered, expected, label=""):
    """Assert element-wise exact match."""
    if sp.issparse(recovered):
        rec = recovered.toarray()
    else:
        rec = np.asarray(recovered)
    if sp.issparse(expected):
        exp = expected.toarray()
    else:
        exp = np.asarray(expected)
    np.testing.assert_array_equal(rec, exp, err_msg=f"Recovery mismatch: {label}")


# ---- Tests: Real PBMC data ----

@pytest.mark.parametrize("dataset", ["pbmc3k.h5ad"])
def test_pbmc_ground_truth(dataset):
    """100% exact recovery on PBMC datasets with ground truth counts."""
    path = PBMC_DIR / dataset
    if not path.exists():
        pytest.skip(f"{path} not found")

    adata = sc.read_h5ad(path)
    result = reverse_log1p(adata.X, base="e")
    _assert_perfect_recovery(result["counts"], adata.layers["counts"], dataset)


# ---- Tests: Simulated data ----

@pytest.mark.parametrize("target_sum", [100, 1e3, 1e4, 1e5, 1e6])
def test_different_target_sum(target_sum):
    """Algorithm is target_sum agnostic."""
    counts = _make_counts()
    log1p = sp.csr_matrix(_normalize_log1p(counts, target_sum=target_sum))
    result = reverse_log1p(log1p, base="e")
    _assert_perfect_recovery(result["counts"], counts, f"target_sum={target_sum}")


@pytest.mark.parametrize("base", ["e", "2", "10"])
def test_different_log_base(base):
    """Works with ln, log2, log10."""
    counts = _make_counts()
    log_data = sp.csr_matrix(_normalize_log1p(counts, base=base))
    result = reverse_log1p(log_data, base=base)
    _assert_perfect_recovery(result["counts"], counts, f"base={base}")


def test_scran_style_size_factors():
    """Works with arbitrary per-cell size factors (not library-size based)."""
    rng = np.random.RandomState(42)
    counts = _make_counts()
    size_factors = rng.lognormal(0, 0.5, size=counts.shape[0])
    normed = counts / size_factors[:, np.newaxis]
    log1p = sp.csr_matrix(np.log1p(normed))
    result = reverse_log1p(log1p, base="e")
    _assert_perfect_recovery(result["counts"], counts, "scran-style")


def test_dense_input():
    """Works with dense arrays."""
    counts = _make_counts(n_cells=50, n_genes=200)
    log1p = _normalize_log1p(counts, target_sum=1e4)
    result = reverse_log1p(log1p, base="e")
    _assert_perfect_recovery(result["counts"], counts, "dense")


def test_robust_min_count_gt_1():
    """Robust mode fixes cells where min nonzero count > 1."""
    counts = _make_counts()
    # Force first 20 cells to have no count=1
    for i in range(20):
        counts[i][counts[i] == 1] = 2
    log1p = sp.csr_matrix(_normalize_log1p(counts, target_sum=1e4))

    result = reverse_log1p(log1p, base="e", robust=True)
    rec = result["counts"].toarray() if sp.issparse(result["counts"]) else result["counts"]
    cell_exact = np.all(rec == counts, axis=1)
    assert cell_exact.sum() == counts.shape[0], (
        f"Robust recovery failed: {cell_exact.sum()}/{counts.shape[0]} cells perfect"
    )


def test_library_sizes_recovered():
    """Recovered library sizes match true library sizes."""
    counts = _make_counts()
    true_lib = counts.sum(axis=1)
    log1p = sp.csr_matrix(_normalize_log1p(counts, target_sum=1e4))
    result = reverse_log1p(log1p, base="e")
    np.testing.assert_array_almost_equal(result["library_sizes"], true_lib, decimal=0)


def test_empty_cells():
    """Handles cells with no expressed genes."""
    counts = _make_counts(n_cells=10, n_genes=1000)
    counts[3, :] = 0  # empty cell
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1  # avoid division by zero
    normed = counts / lib * 1e4
    log1p = sp.csr_matrix(np.log1p(normed))
    result = reverse_log1p(log1p, base="e")
    # Empty cell should remain empty
    rec = result["counts"].toarray() if sp.issparse(result["counts"]) else result["counts"]
    assert rec[3].sum() == 0


# ---- Tests: Detection ----

def test_detect_log1p():
    counts = _make_counts()
    log1p = sp.csr_matrix(_normalize_log1p(counts, target_sum=1e4))
    det = stancounts.detect_normalization(log1p)
    assert det["is_log1p"] is True
    assert det["base"] == "e"


def test_detect_raw_counts():
    counts = sp.csr_matrix(_make_counts())
    det = stancounts.detect_normalization(counts)
    assert det["is_log1p"] is False
    assert det["is_integer"] is True


def test_detect_log2():
    counts = _make_counts()
    log2_data = sp.csr_matrix(_normalize_log1p(counts, base="2"))
    det = stancounts.detect_normalization(log2_data)
    assert det["is_log1p"] is True
    assert det["base"] == "2"


# ---- Tests: AnnData integration ----

def test_anndata_integration():
    counts = _make_counts(n_cells=50, n_genes=200)
    log1p = sp.csr_matrix(_normalize_log1p(counts, target_sum=1e4))

    import anndata
    adata = anndata.AnnData(X=log1p)
    stancounts.reverse_log1p_anndata(adata)

    assert "counts_recovered" in adata.layers
    assert "stancounts_scale_factor" in adata.obs
    _assert_perfect_recovery(adata.layers["counts_recovered"], counts, "anndata")


# ---- Tests: Roundtrip on chondro-atlas (normalize then reverse) ----

def test_chondro_roundtrip():
    """Take a chondro dataset with raw counts, normalize, reverse, compare."""
    # Use the smallest one
    path = CHONDRO_DIR / "01_Ji_2019.h5ad"
    if not path.exists():
        pytest.skip(f"{path} not found")

    adata = sc.read_h5ad(path)
    # .X is already raw counts for this dataset
    counts_true = adata.X
    if sp.issparse(counts_true):
        counts_true = counts_true.toarray().astype(np.float64)
    else:
        counts_true = np.asarray(counts_true, dtype=np.float64)

    # Normalize
    lib = counts_true.sum(axis=1, keepdims=True)
    normed = counts_true / lib * 1e4
    log1p = sp.csr_matrix(np.log1p(normed))

    # Reverse
    result = reverse_log1p(log1p, base="e")
    _assert_perfect_recovery(result["counts"], counts_true, "chondro roundtrip")
