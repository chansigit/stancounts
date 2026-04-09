<p align="center">
  <img src="logo.svg" alt="stancounts logo" width="480">
</p>

<p align="center">
  <strong>Recover raw counts from log1p-normalized single-cell expression matrices</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue?logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="https://github.com/chansigit/stancounts/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://pypi.org/project/stancounts/"><img src="https://img.shields.io/badge/pypi-v0.1.0-orange?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/chansigit/stancounts"><img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey" alt="Platform"></a>
</p>

---

## Why

Many single-cell tools and public datasets distribute only the **log1p-normalized** expression matrix, discarding the raw counts.  Some downstream methods (e.g., RNA velocity, differential expression via negative-binomial models, scVI) *require* integer counts.

`stancounts` reverses the normalization — **exactly**, in one line — without needing to know the original `target_sum`.

## How it works

Standard scRNA-seq normalization:

```
y = log1p(count / library_size × target_sum)
```

The key insight: in every cell, the minimum nonzero value after `expm1` corresponds to a gene with **count = 1** (near-universal in sparse scRNA-seq data). This gives us the per-cell scale factor, and division + rounding recovers exact integer counts.

```
scale_factor = min_nonzero(expm1(y))     # per cell
counts       = round(expm1(y) / scale_factor)
```

The algorithm is **target_sum agnostic** — it works with any normalization target (1e4, 1e6, median, etc.) and any per-cell size factor scheme (scanpy, scran, Seurat).

## Installation

### Python package

```bash
pip install git+https://github.com/chansigit/stancounts.git
```

Or from source:

```bash
git clone https://github.com/chansigit/stancounts.git
cd stancounts
pip install -e .
```

### Claude Code plugin

If you use [Claude Code](https://docs.anthropic.com/en/docs/claude-code), stancounts is also available as a plugin with a built-in skill that guides Claude through the detection and recovery workflow:

```
/plugins add chansigit/stancounts
```

Once installed, simply ask Claude to "recover counts" or "reverse log1p" and the skill activates automatically.

## Quick start

```python
import stancounts
import scanpy as sc

adata = sc.read_h5ad("my_dataset.h5ad")

# One-liner: recover counts into a new layer
stancounts.reverse_log1p_anndata(adata)
# → adata.layers["counts_recovered"]
# → adata.obs["stancounts_scale_factor"]
# → adata.obs["stancounts_library_size"]
```

### Auto-detect normalization

```python
det = stancounts.detect_normalization(adata.X)
print(det)
# {'is_log1p': True, 'base': 'e', 'scores': {'e': 1.0, '2': 0.32, '10': 0.31},
#  'is_integer': False, 'max_value': 7.26}

if det["is_log1p"]:
    stancounts.reverse_log1p_anndata(adata, base=det["base"])
```

### Low-level API

```python
result = stancounts.reverse_log1p(
    adata.X,          # sparse or dense matrix
    base="e",         # "e", "2", or "10"
    robust=True,      # handle rare min_count > 1 cells
)

counts        = result["counts"]          # recovered count matrix
scale_factors = result["scale_factors"]   # per-cell scale factors
library_sizes = result["library_sizes"]   # recovered library sizes
corrections   = result["corrections"]     # robust correction flags
```

## Validation

Tested on **256,000+ real cells** with ground-truth counts:

| Dataset | Cells | Genes | Recovery |
|---------|------:|------:|---------:|
| PBMC3k | 2,638 | 13,714 | **100%** |
| Purified PBMC | 105,753 | 19,512 | **100%** |
| CITE-seq PBMC | 147,582 | 20,729 | **100%** |

All three datasets: **every element, every cell, exactly recovered**.

### Robustness

| Scenario | Result |
|----------|--------|
| `target_sum` = 100, 1e3, 1e4, 1e5, 1e6 | 100% |
| `target_sum` = median(library_size) | 100% |
| scran-style arbitrary size factors | 100% |
| log2(1+x) and log10(1+x) | 100% |
| Cells with min_count > 1 (robust mode) | 100% |
| Deeply sequenced cells (high counts) | 100% |

## API reference

### `stancounts.reverse_log1p(X, *, base="e", robust=True)`

Reverse log1p normalization to recover raw counts.

**Parameters:**
- `X` — expression matrix (cells × genes), sparse or dense
- `base` — log base: `"e"` (default), `"2"`, or `"10"`
- `robust` — detect and correct cells whose minimum nonzero count > 1

**Returns** `dict` with keys: `counts`, `scale_factors`, `library_sizes`, `corrections`

### `stancounts.reverse_log1p_anndata(adata, *, source="X", target_layer="counts_recovered", base="e", robust=True)`

Convenience wrapper for AnnData objects. Stores recovered counts in `adata.layers[target_layer]` and scale factors in `adata.obs`.

### `stancounts.detect_normalization(X, *, n_sample=200, seed=0)`

Auto-detect whether data is log1p-normalized and determine the log base.

**Returns** `dict` with keys: `is_log1p`, `base`, `scores`, `is_integer`, `max_value`

### `stancounts.is_log1p_normalized(X, **kwargs)`

Convenience wrapper: returns `True` if data appears log1p-normalized.

## When it won't work

- **Batch-corrected data** (e.g., Harmony, scVI latent space) — the linear count→normalized relationship is broken
- **scTransform** (Pearson residuals) — not a log1p(scaled_counts) transform
- **Aggregated/pseudobulk** data — may still work if the aggregation preserves the structure
- **Fractional counts** (e.g., from imputation) — no integer counts to recover

## Dependencies

- `numpy >= 1.22`
- `scipy >= 1.7`
- `anndata >= 0.8` *(optional, for AnnData integration)*

## License

MIT
