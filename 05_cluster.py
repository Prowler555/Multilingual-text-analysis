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
    """Read stage-4's manifest.json to get n_sentences / pca_dims."""
    man = json.loads((C.REDUCE_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {"n_sentences": int(man["n_sentences"]), "pca_dims": int(man["pca_dims"])}


def run_umap(args: argparse.Namespace, n_total: int) -> Path:
    """Fit UMAP over the PCA memmap and write the projection to a new memmap.

    UMAP needs the input array in memory, which is why stage 4 reduces to a small
    PCA_DIMS first: (n, pca_dims) float32 is a few hundred MB even at 1.2M rows.
    """
    import numpy as np
    import umap

    pca_dims = load_reduced_meta()["pca_dims"]
    pca = np.memmap(C.REDUCE_DIR / "pca50.f32.memmap", dtype=np.float32, mode="r",
                    shape=(n_total, pca_dims))
    X = np.asarray(pca)  # materialize for UMAP

    LOG.info("fitting UMAP(n_neighbors=%d, min_dist=%.2f, n_components=%d, metric=%s) "
             "on %s x %d ...", args.n_neighbors, args.min_dist, args.n_components,
             args.metric, human_int(n_total), pca_dims)
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        low_memory=C.UMAP_LOW_MEMORY,
        random_state=C.RANDOM_SEED,
    )
    proj = np.asarray(reducer.fit_transform(X), dtype=np.float32)

    out = np.memmap(UMAP_MEMMAP_PATH, dtype=np.float32, mode="w+",
                    shape=(n_total, args.n_components))
    out[:] = proj
    out.flush()
    del out
    LOG.info("wrote UMAP projection -> %s  (%s x %d)",
             UMAP_MEMMAP_PATH, human_int(n_total), args.n_components)
    return UMAP_MEMMAP_PATH


def run_hdbscan_grid(umap_path: Path, n_total: int, args: argparse.Namespace) -> dict:
    """Run HDBSCAN for every entry in C.HDBSCAN_GRID, save labels + a report."""
    import hdbscan
    import numpy as np

    X = np.asarray(np.memmap(umap_path, dtype=np.float32, mode="r",
                             shape=(n_total, args.n_components)))

    entries = []
    for i, params in enumerate(C.HDBSCAN_GRID):
        LOG.info("hdbscan grid %d/%d: %s", i + 1, len(C.HDBSCAN_GRID), params)
        labels = hdbscan.HDBSCAN(**params).fit_predict(X)

        labels_path = C.RESULTS_DIR / f"labels_grid{i}.npy"
        np.save(labels_path, labels)

        n_noise = int((labels == -1).sum())
        _, counts = np.unique(labels[labels != -1], return_counts=True)
        n_clusters = int(counts.size)
        sizes = (
            {"min": int(counts.min()), "median": int(np.median(counts)), "max": int(counts.max())}
            if n_clusters else {"min": 0, "median": 0, "max": 0}
        )
        entry = {
            "index": i,
            "params": params,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_frac": round(n_noise / max(1, n_total), 4),
            "cluster_sizes": sizes,
            "labels_path": labels_path.name,
        }
        entries.append(entry)
        LOG.info("  -> %d clusters, %s noise (%.1f%%), sizes min/med/max %d/%d/%d",
                 n_clusters, human_int(n_noise), 100 * entry["noise_frac"],
                 sizes["min"], sizes["median"], sizes["max"])

    report = {"stage": "05_cluster", "n_sentences": int(n_total), "entries": entries}
    GRID_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", GRID_REPORT_PATH)

    # "Looks best" by cluster count/noise alone - stage 6 does the real silhouette
    # scoring. Prefer the lowest noise fraction among entries with >= 2 clusters.
    candidates = [e for e in entries if e["n_clusters"] >= 2] or entries
    best = min(candidates, key=lambda e: (e["noise_frac"], -e["n_clusters"]))

    return {
        "n_sentences": int(n_total),
        "umap": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "n_components": args.n_components,
            "metric": args.metric,
        },
        "grid_size": len(C.HDBSCAN_GRID),
        "best_grid_index_by_noise": best["index"],
    }


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
