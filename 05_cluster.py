#!/usr/bin/env python3
"""Stage 5 - UMAP + HDBSCAN clustering.

Reads the stage-4 PCA-reduced memmap, projects it further with UMAP
(`UMAP_N_COMPONENTS` dims, cosine metric - good for text embeddings), then
runs HDBSCAN over every `{min_cluster_size, min_samples}` combo in
`HDBSCAN_GRID`. Each grid point gets its own labels file and a line in the
cluster-count/size report so you can pick a setting by inspection before
running stage 6's silhouette score on the winner.

Output:
  data/reduced/umap.f32.memmap        shape (n_sentences, UMAP_N_COMPONENTS)
  results/labels_grid{N}.npy          int labels per grid entry (-1 = noise)
  results/cluster_grid_report.json    cluster count/size per grid entry
  results/manifest.json

Usage:
  python 05_cluster.py                # config.py defaults, full HDBSCAN_GRID
  python 05_cluster.py --dry-run      # print plan, do nothing
  python 05_cluster.py --force        # wipe results/ and start fresh
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import human_int, setup_logging

LOG = setup_logging("05_cluster", C.LOG_DIR)
MANIFEST = C.RESULTS_DIR / "manifest.json"
UMAP_MEMMAP_PATH = C.REDUCE_DIR / "umap.f32.memmap"
GRID_REPORT_PATH = C.RESULTS_DIR / "cluster_grid_report.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-neighbors", type=int, default=C.UMAP_N_NEIGHBORS)
    p.add_argument("--min-dist", type=float, default=C.UMAP_MIN_DIST)
    p.add_argument("--n-components", type=int, default=C.UMAP_N_COMPONENTS)
    p.add_argument("--metric", default=C.UMAP_METRIC)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_reduced_meta() -> dict:
    """Read stage-4's manifest.json to get n_sentences / pca_dims.

    TODO: json.loads(C.REDUCE_DIR / "manifest.json").
    """
    raise NotImplementedError


def run_umap(args: argparse.Namespace, n_total: int) -> Path:
    """Fit UMAP over the PCA memmap and write the projection to a new memmap.

    TODO:
      - open C.REDUCE_DIR / "pca50.f32.memmap" read-only with shape from
        load_reduced_meta()
      - umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        n_components=args.n_components, metric=args.metric,
        low_memory=C.UMAP_LOW_MEMORY, random_state=C.RANDOM_SEED).fit_transform(...)
        (note: UMAP wants the array in memory, so this is the point where the
        pipeline needs PCA_DIMS to already be small - keep an eye on RAM here)
      - write the result to UMAP_MEMMAP_PATH
    Returns the path written.
    """
    raise NotImplementedError


def run_hdbscan_grid(umap_path: Path, n_total: int, args: argparse.Namespace) -> dict:
    """Run HDBSCAN for every entry in C.HDBSCAN_GRID, save labels + a report.

    TODO:
      - open umap_path memmap read-only
      - for each {min_cluster_size, min_samples} in C.HDBSCAN_GRID:
          hdbscan.HDBSCAN(**params).fit_predict(X)
          save labels to results/labels_grid{i}.npy
          record n_clusters, n_noise, cluster size histogram (min/median/max)
      - write GRID_REPORT_PATH with one entry per grid point
    Returns a dict of stats for the manifest (e.g. which grid index looked best
    by cluster count alone - stage 6 does the real scoring).
    """
    raise NotImplementedError


def write_manifest(stats: dict, t0: float) -> None:
    manifest = {
        "stage": "05_cluster",
        **stats,
        "umap_memmap_path": str(UMAP_MEMMAP_PATH),
        "grid_report_path": str(GRID_REPORT_PATH),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", MANIFEST)


def main() -> None:
    args = parse_args()
    reduce_manifest_path = C.REDUCE_DIR / "manifest.json"

    LOG.info("umap: n_neighbors=%d min_dist=%.2f n_components=%d metric=%s | grid=%d combos",
              args.n_neighbors, args.min_dist, args.n_components, args.metric,
              len(C.HDBSCAN_GRID))

    if not reduce_manifest_path.exists():
        raise SystemExit(f"No input found in {C.REDUCE_DIR} - run 04_reduce.py first.")

    if args.dry_run:
        LOG.info("dry-run: would UMAP-project then run HDBSCAN over %d grid combo(s): %s",
                  len(C.HDBSCAN_GRID), C.HDBSCAN_GRID)
        return

    if GRID_REPORT_PATH.exists() and not args.force:
        raise SystemExit(f"{GRID_REPORT_PATH} already exists. Pass --force to overwrite.")
    if args.force:
        for p in C.RESULTS_DIR.glob("labels_grid*.npy"):
            p.unlink()
        GRID_REPORT_PATH.unlink(missing_ok=True)
        UMAP_MEMMAP_PATH.unlink(missing_ok=True)
        MANIFEST.unlink(missing_ok=True)

    reduced_meta = load_reduced_meta()
    n_total = reduced_meta["n_sentences"]
    LOG.info("%s sentences to cluster", human_int(n_total))

    t0 = time.time()
    umap_path = run_umap(args, n_total)
    stats = run_hdbscan_grid(umap_path, n_total, args)
    write_manifest(stats, t0)
    LOG.info("DONE - see %s", MANIFEST)


if __name__ == "__main__":
    main()
