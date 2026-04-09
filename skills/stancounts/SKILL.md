---
name: stancounts
description: >
  Use when the user asks to "reverse log1p", "recover counts",
  "undo normalization", "convert normalized to counts",
  "reverse normalization", "get raw counts from log1p",
  "infer library size", "denormalize expression matrix",
  or when working with single-cell transcriptomics data that
  has been log1p-normalized and needs to be reverted to raw counts.
  Do NOT use for gene harmonization (that is stangene) or
  format conversion (that is stanobj).
version: 1.0.0
allowed-tools: [Bash, Read, Glob, Grep]
---

# Skill: Reverse Log1p Normalization to Recover Raw Counts

Use this skill when the user asks to recover raw counts from a log1p-normalized
single-cell expression matrix. The algorithm infers per-cell scale factors from
the minimum nonzero value after expm1, enabling exact count recovery without
knowing the original target_sum.

## Prerequisites

1. Check if stancounts is installed:
   ```bash
   python -c "import stancounts; print(stancounts.__version__)"
   ```
   If not installed:
   ```bash
   pip install git+https://github.com/chansigit/stancounts.git
   ```

## Quick Usage

### Basic: Reverse a matrix

```python
import stancounts

result = stancounts.reverse_log1p(adata.X)
counts = result["counts"]          # recovered count matrix (sparse)
scale_factors = result["scale_factors"]  # per-cell normalization factors
library_sizes = result["library_sizes"]  # recovered library sizes
```

### Auto-detect and reverse

```python
import stancounts

det = stancounts.detect_normalization(adata.X)
# det = {"is_log1p": True, "base": "e", "scores": {...}, "is_integer": False, "max_value": 7.26}

if det["is_log1p"]:
    result = stancounts.reverse_log1p(adata.X, base=det["base"])
```

### AnnData integration

```python
import stancounts
import scanpy as sc

adata = sc.read_h5ad("dataset.h5ad")
stancounts.reverse_log1p_anndata(adata)
# Adds: adata.layers["counts_recovered"]
#        adata.obs["stancounts_scale_factor"]
#        adata.obs["stancounts_library_size"]
```

## Parameters

### `reverse_log1p(X, *, base="e", robust=True)`
- **X**: log1p-normalized expression matrix (cells x genes), sparse or dense
- **base**: Log base used in normalization (`"e"`, `"2"`, or `"10"`)
- **robust**: If True, detect and correct cells with min_count > 1

### `detect_normalization(X, *, n_sample=200, seed=0)`
- Auto-detects if data is log1p-normalized
- Determines the log base (ln, log2, log10)
- Distinguishes from raw integer counts

### `reverse_log1p_anndata(adata, *, source="X", target_layer="counts_recovered", base="e", robust=True)`
- Convenience wrapper that stores results in AnnData layers and obs

## Workflow

1. **Detect** whether the data is log1p-normalized:
   ```python
   det = stancounts.detect_normalization(adata.X)
   ```
   Report the detection result to the user.

2. **If already raw counts** (`det["is_log1p"] == False` and `det["is_integer"] == True`):
   Tell the user the data appears to already be raw counts.

3. **If log1p-normalized** (`det["is_log1p"] == True`):
   ```python
   stancounts.reverse_log1p_anndata(adata, base=det["base"])
   ```

4. **Report** the recovery results:
   - Number of cells and genes
   - Recovered library size stats (min, median, max)
   - Whether robust corrections were needed
   - Save the modified AnnData if requested

## Important

- The algorithm is **target_sum agnostic** — works with any normalization target
- Works with scanpy, scran, Seurat-style normalization
- 100% exact recovery validated on 256k+ real cells
- Only fails if a cell has NO gene with count=1 (extremely rare, <0.001%)
- The `robust=True` mode handles even this edge case
