"""Obtain integer counts from any AnnData: existing layer, X, .raw, or recover."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from stancounts.core import reverse_log1p
from stancounts.detect import detect_normalization

DEFAULT_PREFER_LAYERS = (
    "counts", "count", "raw_counts", "counts_raw",
    "umi", "umis", "umi_counts", "X_counts",
)
DEFAULT_EXCLUDE_LAYERS = (
    "spliced", "unspliced", "ambiguous",
    "spliced_counts", "unspliced_counts", "matrix",
)


class CountsUnavailable(ValueError):
    """Integer counts could not be found in layers/X/.raw nor recovered from X."""


def _sample_idx(n_rows: int, n_sample: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if n_rows <= n_sample:
        return np.arange(n_rows)
    return rng.choice(n_rows, n_sample, replace=False)


def _is_integer_matrix(M, *, n_sample: int = 200, seed: int = 0) -> bool:
    """True if sampled nonzero values are finite, non-negative, near-integer."""
    if M is None:
        return False
    idx = _sample_idx(M.shape[0], n_sample, seed)
    if sp.issparse(M):
        csr = M.tocsr()
        parts = [csr.data[csr.indptr[i]:csr.indptr[i + 1]] for i in idx]
        data = np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        sub = np.asarray(M[idx])
        data = sub[sub != 0].ravel()
    if data.size == 0:
        return False
    data = data.astype(np.float64)
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        return False
    return bool(np.allclose(data, np.round(data), rtol=0, atol=1e-6))


# Layer-name substrings hinting at log-normalized data; such layers are tried
# first when recovering counts from a layer (X is always tried before any layer).
_RECOVERY_NAME_HINTS = ("lognorm", "normalized", "data")


def _recovery_candidates(adata, exclude):
    """Yield (source_label, matrix) to try for log1p recovery.

    X first (label ``"recovered"``, preserving the historical source string), then
    non-excluded layers whose name hints at normalized data, then the remaining
    layers in natural order. This lets a scaled/negative X — which is neither
    integer nor log1p — fall back to a log1p-normalized layer (e.g. ``lognorm``).
    """
    yield "recovered", adata.X
    layers = [n for n in adata.layers if n not in exclude]
    hinted = [n for n in layers if any(h in n.lower() for h in _RECOVERY_NAME_HINTS)]
    rest = [n for n in layers if n not in hinted]
    for name in hinted + rest:
        yield f"recovered:layer:{name}", adata.layers[name]


def get_counts(
    adata,
    *,
    prefer_layers=DEFAULT_PREFER_LAYERS,
    exclude_layers=DEFAULT_EXCLUDE_LAYERS,
    base: str = "e",
    robust: bool = True,
    allow_recovery: bool = True,
    n_sample: int = 200,
    seed: int = 0,
) -> dict:
    """Return integer counts (aligned to adata.var_names) from any AnnData.

    Priority: whitelist integer layer -> integer X -> integer .raw (aligned)
    -> reverse_log1p(X) if X is log1p. Raises CountsUnavailable otherwise.
    Velocity layers (spliced/unspliced/...) are never treated as counts.

    Note: the `base` parameter is currently unused — when recovering from
    log1p, the base is auto-detected via `detect_normalization` (``det["base"]``).
    """
    exclude = set(exclude_layers)

    # 1. whitelist layers, in order
    for name in prefer_layers:
        if name in exclude or name not in adata.layers:
            continue
        M = adata.layers[name]
        if _is_integer_matrix(M, n_sample=n_sample, seed=seed):
            return {"counts": M, "source": f"layer:{name}"}

    # 2. X is integer
    if _is_integer_matrix(adata.X, n_sample=n_sample, seed=seed):
        return {"counts": adata.X, "source": "X"}

    # 3. .raw, aligned to adata.var_names (raw must cover all genes)
    raw = adata.raw
    if raw is not None and _is_integer_matrix(raw.X, n_sample=n_sample, seed=seed):
        raw_pos = {g: i for i, g in enumerate(raw.var_names)}
        if all(g in raw_pos for g in adata.var_names):
            cols = [raw_pos[g] for g in adata.var_names]
            rawX = raw.X
            aligned = rawX.tocsc()[:, cols].tocsr() if sp.issparse(rawX) else np.asarray(rawX)[:, cols]
            return {"counts": aligned, "source": "raw"}

    # 4. recover from log1p-normalized X, then from a log1p-normalized layer
    #    (so a scaled/negative X falls back to e.g. layers["lognorm"]).
    if allow_recovery:
        for src, M in _recovery_candidates(adata, exclude):
            if M is None:
                continue
            det = detect_normalization(M, n_sample=n_sample, seed=seed)
            if det["is_log1p"]:
                rec = reverse_log1p(M, base=det["base"], robust=robust)
                return {"counts": rec["counts"], "source": src, "base": det["base"]}

    raise CountsUnavailable(
        "no integer counts in whitelist layers / X / .raw, and neither X nor any "
        "layer is log1p-recoverable"
    )
