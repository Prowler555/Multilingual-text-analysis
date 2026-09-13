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
    """Read stage-5's cluster_grid_report.json.

    TODO: json.loads(GRID_REPORT_PATH).
    """
    raise NotImplementedError


def pick_grid_entry(report: dict, grid_index: int | None) -> int:
    """Choose which grid entry to score.

    TODO: if grid_index is given, use it; otherwise pick a reasonable default
    (e.g. fewest noise points among entries with a sane cluster count, not
    just "most clusters" which HDBSCAN can game with tiny min_cluster_size).
    """
    raise NotImplementedError


def compute_silhouette(grid_index: int, sample_size: int) -> float:
    """Silhouette score on a fixed random sample of the chosen grid entry.

    TODO:
      - load results/labels_grid{grid_index}.npy and the UMAP or PCA memmap
        the labels were computed on (use the same space as stage 5)
      - drop noise points (label == -1) before sampling, or keep them and
        note it explicitly - decide once and document it here
      - np.random.RandomState(C.RANDOM_SEED).choice(..., size=sample_size)
      - sklearn.metrics.silhouette_score(X_sample, labels_sample)
    """
    raise NotImplementedError


def gather_dataset_stats() -> dict:
    """Pull real counts from every prior stage's manifest.json for the README.

    TODO: read manifest.json from RAW_DIR, SENT_DIR, EMB_DIR to report
    dataset size (stage 1), clean sentence count (stage 2), etc. - no
    placeholders, only numbers a manifest actually recorded.
    """
    raise NotImplementedError


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
    silhouette = compute_silhouette(grid_index, args.silhouette_sample)
    dataset_stats = gather_dataset_stats()

    stats = {
        "stage": "06_evaluate",
        "grid_index": grid_index,
        "grid_params": C.HDBSCAN_GRID[grid_index],
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
