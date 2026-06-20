"""Tests for stancounts.get_counts (source selection + recovery)."""

import anndata
import numpy as np
import pytest
import scipy.sparse as sp

import stancounts
from stancounts import CountsUnavailable, get_counts


def _counts(n_cells=120, n_genes=60, seed=0):
    rng = np.random.RandomState(seed)
    return rng.poisson(0.6, size=(n_cells, n_genes)).astype(np.float64)


def _lognorm(counts, target_sum=1e4):
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    return np.log1p(counts / lib * target_sum)


def test_from_counts_layer():
    counts = _counts()
    ad = anndata.AnnData(X=_lognorm(counts))
    ad.layers["counts"] = sp.csr_matrix(counts)
    res = get_counts(ad)
    assert res["source"] == "layer:counts"
    np.testing.assert_array_equal(np.asarray(res["counts"].todense()), counts)


def test_whitelist_beats_X():
    counts = _counts()
    ad = anndata.AnnData(X=sp.csr_matrix(counts))  # X is integer too
    ad.layers["raw_counts"] = sp.csr_matrix(counts)
    res = get_counts(ad)
    assert res["source"] == "layer:raw_counts"  # whitelist checked before X


def test_from_X_integer():
    counts = _counts()
    ad = anndata.AnnData(X=sp.csr_matrix(counts))
    res = get_counts(ad)
    assert res["source"] == "X"
    np.testing.assert_array_equal(np.asarray(res["counts"].todense()), counts)


def test_from_raw_aligned():
    counts = _counts(n_genes=60)
    ad = anndata.AnnData(X=_lognorm(counts[:, :40]))  # adata has 40 genes
    ad.var_names = [f"G{i}" for i in range(40)]
    raw = anndata.AnnData(X=sp.csr_matrix(counts))    # raw has 60 genes (superset)
    raw.var_names = [f"G{i}" for i in range(60)]
    ad.raw = raw
    res = get_counts(ad)
    assert res["source"] == "raw"
    assert res["counts"].shape == (counts.shape[0], 40)
    np.testing.assert_array_equal(np.asarray(res["counts"].todense()), counts[:, :40])


def test_recovered_from_log1p():
    counts = _counts()
    ad = anndata.AnnData(X=sp.csr_matrix(_lognorm(counts)))  # only log1p X, no counts
    res = get_counts(ad)
    assert res["source"] == "recovered"
    np.testing.assert_array_equal(np.asarray(res["counts"].todense()), counts)


def test_velocity_layers_not_mistaken_for_counts():
    counts = _counts()
    ad = anndata.AnnData(X=sp.csr_matrix(_lognorm(counts)))
    ad.layers["spliced"] = sp.csr_matrix(_counts(seed=1))    # integer, but velocity
    ad.layers["unspliced"] = sp.csr_matrix(_counts(seed=2))
    res = get_counts(ad)
    assert res["source"] == "recovered"  # spliced/unspliced excluded -> recover from X


def test_unavailable_raises():
    rng = np.random.RandomState(0)
    floats = rng.uniform(0.1, 5.0, size=(50, 30))  # non-integer, not log1p of counts
    ad = anndata.AnnData(X=floats)
    with pytest.raises(CountsUnavailable):
        get_counts(ad, allow_recovery=False)


def test_custom_prefer_and_exclude_override():
    counts = _counts()
    ad = anndata.AnnData(X=sp.csr_matrix(_lognorm(counts)))
    ad.layers["my_counts"] = sp.csr_matrix(counts)
    # "my_counts" not in default whitelist -> would recover; custom prefer picks it
    res = get_counts(ad, prefer_layers=("my_counts",))
    assert res["source"] == "layer:my_counts"
    # excluding it -> falls back to recovery from X
    res2 = get_counts(ad, prefer_layers=("my_counts",), exclude_layers=("my_counts",))
    assert res2["source"] == "recovered"
