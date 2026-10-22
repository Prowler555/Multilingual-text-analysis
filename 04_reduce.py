#!/usr/bin/env python3
"""Stage 4 - dimensionality reduction (PCA).

Reads the stage-3 embedding memmap (n_sentences x EMB_DIM, float16) and
reduces it to `PCA_DIMS` dimensions with scikit-learn's IncrementalPCA, fit
and transformed in chunks (`PCA_BATCH` rows at a time) so the full matrix
never has to sit in RAM as float32 at once. This is a cheap, linear
pre-reduction before UMAP does the real nonlinear reduction in stage 5.

Output:
  data/reduced/pca50.f32.memmap   shape (n_sentences, PCA_DIMS), dtype float32
  data/reduced/pca_model.joblib   fitted IncrementalPCA (for reuse/inspection)
  data/reduced/manifest.json

Usage:
  python 04_reduce.py                 # config.py defaults
  python 04_reduce.py --dry-run       # print plan, do nothing
  python 04_reduce.py --force         # wipe data/reduced and start fresh
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import human_int, setup_logging

LOG = setup_logging("04_reduce", C.LOG_DIR)
MANIFEST = C.REDUCE_DIR / "manifest.json"
MEMMAP_PATH = C.REDUCE_DIR / "pca50.f32.memmap"
MODEL_PATH = C.REDUCE_DIR / "pca_model.joblib"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pca-dims", type=int, default=C.PCA_DIMS)
    p.add_argument("--batch-size", type=int, default=C.PCA_BATCH)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_embeddings_meta() -> dict:
    """Read stage-3's manifest.json to get n_sentences / dim without opening the memmap.

    Stage 3 records the embedding width as `emb_dim` (taken from the model, not
    the config constant); we expose it here as `dim`.
    """
    man = json.loads((C.EMB_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {"n_sentences": int(man["n_sentences"]), "dim": int(man["emb_dim"])}


def _fit_chunks(n_total: int, bs: int, k: int):
    """Yield (start, end) fit-chunk bounds, each with >= k rows.

    IncrementalPCA.partial_fit requires n_samples >= n_components per call, so a
    trailing chunk shorter than k is folded into the preceding one.
    """
    start = 0
    while start < n_total:
        end = min(start + bs, n_total)
        if 0 < n_total - end < k:  # absorb a too-short remainder now
            end = n_total
        yield start, end
        start = end


def fit_incremental_pca(emb_memmap, n_total: int, args: argparse.Namespace) -> dict:
    """Two-pass IncrementalPCA: partial_fit over chunks, then transform chunks.

    `emb_memmap` is unused (main passes None) - the stage-3 memmap is opened here
    read-only so this function owns the whole PCA lifecycle.
    """
    import joblib
    import numpy as np
    from sklearn.decomposition import IncrementalPCA

    meta = load_embeddings_meta()
    dim, k, bs = meta["dim"], args.pca_dims, args.batch_size
    if n_total < k:
        raise SystemExit(f"need at least {k} sentences for {k}-dim PCA, have {n_total}")

    emb = np.memmap(C.EMB_DIR / "emb.f16.memmap", dtype=np.float16, mode="r",
                    shape=(n_total, dim))
    pca = IncrementalPCA(n_components=k)

    # Pass 1: fit. float16 -> float32 per chunk so the full matrix never lands in RAM.
    LOG.info("pass 1/2: fitting IncrementalPCA(%d) over %s rows (%d -> %d) ...",
             k, human_int(n_total), dim, k)
    last_log = time.time()
    for start, end in _fit_chunks(n_total, bs, k):
        pca.partial_fit(np.asarray(emb[start:end], dtype=np.float32))
        if time.time() - last_log >= 5.0:
            LOG.info("  fit %s / %s", human_int(end), human_int(n_total))
            last_log = time.time()

    # Pass 2: transform into the output memmap.
    out = np.memmap(MEMMAP_PATH, dtype=np.float32, mode="w+", shape=(n_total, k))
    LOG.info("pass 2/2: transforming -> %s", MEMMAP_PATH)
    last_log = time.time()
    start = 0
    while start < n_total:
        end = min(start + bs, n_total)
        out[start:end] = pca.transform(np.asarray(emb[start:end], dtype=np.float32))
        start = end
        if time.time() - last_log >= 5.0:
            LOG.info("  transform %s / %s", human_int(end), human_int(n_total))
            last_log = time.time()
    out.flush()
    del out

    joblib.dump(pca, MODEL_PATH)
    evr = pca.explained_variance_ratio_
    evr_sum = float(evr.sum())
    LOG.info("saved %s; explained variance (sum of %d comps) = %.4f", MODEL_PATH, k, evr_sum)

    return {
        "n_sentences": int(n_total),
        "input_dim": int(dim),
        "pca_dims": int(k),
        "batch_size": int(bs),
        "explained_variance_ratio_sum": evr_sum,
        "dtype": "float32",
    }


def write_manifest(stats: dict, t0: float) -> None:
    manifest = {
        "stage": "04_reduce",
        **stats,
        "memmap_path": str(MEMMAP_PATH),
        "model_path": str(MODEL_PATH),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", MANIFEST)


def main() -> None:
    args = parse_args()
    emb_manifest_path = C.EMB_DIR / "manifest.json"

    LOG.info("pca_dims=%d  batch_size=%d", args.pca_dims, args.batch_size)

    if not emb_manifest_path.exists():
        raise SystemExit(f"No input found in {C.EMB_DIR} - run 03_embed.py first.")

    if args.dry_run:
        LOG.info("dry-run: would fit IncrementalPCA(%d) over %s in chunks of %s rows -> %s",
                  args.pca_dims, C.EMB_DIR / "emb.f16.memmap",
                  human_int(args.batch_size), MEMMAP_PATH)
        return

    if MEMMAP_PATH.exists() and not args.force:
        raise SystemExit(f"{MEMMAP_PATH} already exists. Pass --force to overwrite.")
    if args.force and MEMMAP_PATH.exists():
        LOG.warning("--force: removing existing reduced embeddings")
        MEMMAP_PATH.unlink()
        MODEL_PATH.unlink(missing_ok=True)
        MANIFEST.unlink(missing_ok=True)

    emb_meta = load_embeddings_meta()
    n_total = emb_meta["n_sentences"]
    LOG.info("%s sentences to reduce %d -> %d dims",
              human_int(n_total), emb_meta["dim"], args.pca_dims)

    t0 = time.time()
    stats = fit_incremental_pca(None, n_total, args)
    write_manifest(stats, t0)
    LOG.info("DONE - see %s", MANIFEST)


if __name__ == "__main__":
    main()
