"""Test stancounts on chondro-atlas datasets."""

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

import stancounts

DATA_DIR = Path("/home/users/chensj16/s/data/chondro-atlas/h5ad")

h5ad_files = sorted(DATA_DIR.glob("*.h5ad"))
print(f"Found {len(h5ad_files)} h5ad files\n")

for f in h5ad_files:
    print(f"{'='*70}")
    print(f"  {f.name}")
    print(f"{'='*70}")

    adata = sc.read_h5ad(f, backed="r")
    print(f"  Shape: {adata.shape}")
    print(f"  X dtype: {adata.X.dtype}")
    print(f"  Layers: {list(adata.layers.keys())}")

    # Check what's in .X
    X = adata.X
    if hasattr(X, 'dtype'):
        print(f"  X type: {type(X).__name__}")

    # Check a sample of values
    try:
        sample = adata.X[:5]
        if sp.issparse(sample):
            sample_dense = sample.toarray()
        else:
            sample_dense = np.asarray(sample)
    except Exception as e:
        print(f"  Error reading sample: {e}")
        adata.file.close()
        continue

    nz = sample_dense[sample_dense > 0]
    if len(nz) > 0:
        print(f"  Sample nonzero stats: min={nz.min():.4f}, max={nz.max():.4f}, "
              f"mean={nz.mean():.4f}")
        is_int = np.allclose(nz, np.round(nz), atol=1e-4)
        print(f"  Values are integer-like: {is_int}")
    else:
        print(f"  No nonzero values in sample")

    # Check if counts layer exists (ground truth)
    has_counts = "counts" in adata.layers
    has_raw_counts = "raw_counts" in adata.layers
    print(f"  Has 'counts' layer: {has_counts}")
    print(f"  Has 'raw_counts' layer: {has_raw_counts}")

    # Detect normalization
    try:
        # Read a chunk for detection (backed mode)
        chunk = adata.X[:min(500, adata.n_obs)]
        if sp.issparse(chunk):
            chunk = chunk.tocsr()

        det = stancounts.detect_normalization(chunk)
        print(f"  Detection: is_log1p={det['is_log1p']}, base={det['base']}, "
              f"scores={det['scores']}")
        print(f"  is_integer={det['is_integer']}, max_value={det['max_value']:.2f}")

        if det["is_log1p"]:
            # Try recovery
            result = stancounts.reverse_log1p(chunk, base=det["base"])
            counts_rec = result["counts"]
            print(f"  Recovered counts: type={type(counts_rec).__name__}, "
                  f"dtype={counts_rec.dtype}")

            if sp.issparse(counts_rec):
                rec_dense = counts_rec[:5].toarray()
            else:
                rec_dense = counts_rec[:5]

            nz_rec = rec_dense[rec_dense > 0]
            if len(nz_rec) > 0:
                print(f"  Recovered nonzero: min={nz_rec.min():.1f}, "
                      f"max={nz_rec.max():.1f}, mean={nz_rec.mean():.2f}")
                is_int_rec = np.allclose(nz_rec, np.round(nz_rec), atol=1e-3)
                print(f"  Recovered are integers: {is_int_rec}")

            # If ground truth exists, compare
            if has_counts:
                gt = adata.layers["counts"][:min(500, adata.n_obs)]
                if sp.issparse(gt):
                    gt_dense = gt.toarray()
                else:
                    gt_dense = np.asarray(gt)

                if sp.issparse(counts_rec):
                    rec_full = counts_rec.toarray()
                else:
                    rec_full = np.asarray(counts_rec)

                exact = (rec_full == gt_dense)
                nz_mask = gt_dense > 0
                print(f"  vs ground truth:")
                print(f"    Element-wise exact: {exact.sum()}/{exact.size} "
                      f"({exact.sum()/exact.size*100:.2f}%)")
                if nz_mask.sum() > 0:
                    exact_nz = (rec_full[nz_mask] == gt_dense[nz_mask])
                    print(f"    Nonzero exact: {exact_nz.sum()}/{nz_mask.sum()} "
                          f"({exact_nz.sum()/nz_mask.sum()*100:.2f}%)")
        else:
            print(f"  Data does not appear log1p-normalized, skipping recovery")

    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()

    adata.file.close()
    print()
