#!/usr/bin/env python3
"""Stage 6 - evaluation.

Picks a grid entry from stage 5's report (best by default, or `--grid-index`
to force one), computes silhouette score on a fixed random sample
(`SILHOUETTE_SAMPLE` rows - silhouette is O(n^2), so scoring the full set
isn't an option), and writes `results/metrics.json`. Also rewrites the
`## Results` block in README.md between the `<!-- RESULTS:START/END -->`
markers so the README always reflects a real run, never a hand-edited guess.

Output:
  results/metrics.json
  README.md (Results section rewritten in place)

Usage:
  python 06_evaluate.py                    # pick best grid entry automatically
  python 06_evaluate.py --grid-index 2     # score a specific HDBSCAN_GRID entry
  python 06_evaluate.py --dry-run          # print plan, do nothing
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import human_int, setup_logging

LOG = setup_logging("06_evaluate", C.LOG_DIR)
METRICS_PATH = C.RESULTS_DIR / "metrics.json"
GRID_REPORT_PATH = C.RESULTS_DIR / "cluster_grid_report.json"
STAGE5_MANIFEST = C.RESULTS_DIR / "manifest.json"
UMAP_MEMMAP_PATH = C.REDUCE_DIR / "umap.f32.memmap"
README_PATH = C.ROOT / "README.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--grid-index", type=int, default=None,
                   help="score this HDBSCAN_GRID entry instead of auto-picking one")
    p.add_argument("--silhouette-sample", type=int, default=C.SILHOUETTE_SAMPLE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_grid_report() -> dict:
    """Read stage-5's cluster_grid_report.json."""
    return json.loads(GRID_REPORT_PATH.read_text(encoding="utf-8"))


def pick_grid_entry(report: dict, grid_index: int | None) -> int:
    """Choose which grid entry to score.

    Explicit --grid-index wins. Otherwise pick the lowest noise fraction among
    entries with >= 2 clusters (tie-break: more clusters). This avoids rewarding
    a tiny min_cluster_size that "gams" the count with many micro-clusters.
    """
    entries = report["entries"]
    if grid_index is not None:
        if not (0 <= grid_index < len(entries)):
            raise SystemExit(f"--grid-index {grid_index} out of range (0..{len(entries) - 1})")
        return grid_index

    candidates = [e for e in entries if e["n_clusters"] >= 2] or entries
    best = min(candidates, key=lambda e: (e["noise_frac"], -e["n_clusters"]))
    LOG.info("auto-picked grid %d (%s): %d clusters, %.1f%% noise",
             best["index"], best["params"], best["n_clusters"], 100 * best["noise_frac"])
    return int(best["index"])


def compute_silhouette(grid_index: int, sample_size: int) -> float:
    """Silhouette score on a fixed random sample of the chosen grid entry.

    Scored in the SAME space stage 5 clustered in (the UMAP projection), with
    euclidean distance. Noise points (label == -1) are dropped first - silhouette
    is undefined for the catch-all noise "cluster".
    """
    import numpy as np
    from sklearn.metrics import silhouette_score

    labels = np.load(C.RESULTS_DIR / f"labels_grid{grid_index}.npy")
    man = json.loads(STAGE5_MANIFEST.read_text(encoding="utf-8"))
    n_total = int(man["n_sentences"])
    n_comp = int(man["umap"]["n_components"])
    X = np.asarray(np.memmap(UMAP_MEMMAP_PATH, dtype=np.float32, mode="r",
                             shape=(n_total, n_comp)))

    keep = labels != -1
    X_valid, labels_valid = X[keep], labels[keep]
    n_valid = int(labels_valid.size)
    if np.unique(labels_valid).size < 2:
        LOG.warning("fewer than 2 non-noise clusters - silhouette undefined (nan)")
        return float("nan")

    rng = np.random.RandomState(C.RANDOM_SEED)
    if n_valid > sample_size:
        idx = rng.choice(n_valid, size=sample_size, replace=False)
        X_s, labels_s = X_valid[idx], labels_valid[idx]
        if np.unique(labels_s).size < 2:  # unlucky sample collapsed to one cluster
            LOG.warning("sample collapsed to one cluster - scoring all non-noise points")
            X_s, labels_s = X_valid, labels_valid
    else:
        X_s, labels_s = X_valid, labels_valid

    LOG.info("silhouette on %s of %s non-noise points",
             human_int(labels_s.size), human_int(n_valid))
    return float(silhouette_score(X_s, labels_s, metric="euclidean"))


def gather_dataset_stats() -> dict:
    """Pull real counts from every prior stage's manifest.json for the README.

    Only numbers a manifest actually recorded - missing manifests degrade to
    'n/a'/0 rather than invented placeholders.
    """
    def _load(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    s1 = _load(C.RAW_DIR / "manifest.json")
    s2 = _load(C.SENT_DIR / "manifest.json")
    s3 = _load(C.EMB_DIR / "manifest.json")
    return {
        "dataset_size_human": s1.get("text_human", "n/a"),
        "dataset_text_bytes": s1.get("text_bytes"),
        "articles_total": s1.get("articles_total"),
        "languages": s1.get("languages"),
        "clean_sentences": int(s2.get("sentences_kept", 0)),
        "embedded_sentences": s3.get("n_sentences"),
    }


def write_metrics(stats: dict) -> None:
    METRICS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", METRICS_PATH)


def update_readme(stats: dict) -> None:
    """Rewrite the block between <!-- RESULTS:START --> and <!-- RESULTS:END -->."""
    block = (
        "<!-- RESULTS:START -->\n"
        f"- Dataset size: {stats['dataset_size_human']}\n"
        f"- Clean sentences: {human_int(stats['clean_sentences'])}\n"
        f"- Clusters found: {human_int(stats['n_clusters'])}\n"
        f"- Silhouette score: {stats['silhouette_score']:.3f}\n"
        "<!-- RESULTS:END -->"
    )
    text = README_PATH.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
        block, text, flags=re.DOTALL,
    )
    if n == 0:
        raise SystemExit("README.md is missing the RESULTS:START/END markers.")
    README_PATH.write_text(new_text, encoding="utf-8")
    LOG.info("updated Results section in %s", README_PATH)


def main() -> None:
    args = parse_args()

    LOG.info("grid_index=%s  silhouette_sample=%s",
              args.grid_index if args.grid_index is not None else "auto",
              human_int(args.silhouette_sample))

    if not GRID_REPORT_PATH.exists():
        raise SystemExit(f"{GRID_REPORT_PATH} not found - run 05_cluster.py first.")

    if args.dry_run:
        LOG.info("dry-run: would score grid entry %s on a %s-row silhouette sample, "
                  "then update %s and %s",
                  args.grid_index if args.grid_index is not None else "(auto-picked)",
                  human_int(args.silhouette_sample), METRICS_PATH, README_PATH)
        return

    t0 = time.time()
    report = load_grid_report()
    grid_index = pick_grid_entry(report, args.grid_index)
    n_clusters = int(report["entries"][grid_index]["n_clusters"])
    silhouette = compute_silhouette(grid_index, args.silhouette_sample)
    dataset_stats = gather_dataset_stats()

    stats = {
        "stage": "06_evaluate",
        "grid_index": grid_index,
        "grid_params": C.HDBSCAN_GRID[grid_index],
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "silhouette_sample": args.silhouette_sample,
        **dataset_stats,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_metrics(stats)
    update_readme(stats)
    LOG.info("DONE - see %s", METRICS_PATH)


if __name__ == "__main__":
    main()
