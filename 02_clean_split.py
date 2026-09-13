#!/usr/bin/env python3
"""Stage 2 - cleaning & sentence splitting.

Reads the raw JSONL shards from stage 1, splits each article into sentences
with a blank spaCy pipeline + sentencizer (multiprocessing via `nlp.pipe`),
drops junk (too short/long, digit-heavy, boilerplate), and writes clean
sentences out as parquet shards. Stops once `TARGET_SENTENCES` is reached so
later stages have a fixed, known-size input.

Design constraints (match stage 1):
  * Stream shard-by-shard, never load all of data/raw/ into RAM at once.
  * Multiprocess the spaCy pipeline (`N_PROCESS` workers) since sentence
    splitting is CPU-bound and this is the slowest stage on a laptop CPU.

Output:
  data/sentences/sent_00000.parquet, ...   columns: id, lang, sentence
  data/sentences/manifest.json

Usage:
  python 02_clean_split.py                     # config.py defaults
  python 02_clean_split.py --target-sentences 200000   # fast iteration slice
  python 02_clean_split.py --dry-run            # print plan, do nothing
  python 02_clean_split.py --force              # wipe data/sentences and start fresh
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import human_int, setup_logging

LOG = setup_logging("02_clean_split", C.LOG_DIR)
MANIFEST = C.SENT_DIR / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target-sentences", type=int, default=C.TARGET_SENTENCES)
    p.add_argument("--min-chars", type=int, default=C.SENT_MIN_CHARS)
    p.add_argument("--max-chars", type=int, default=C.SENT_MAX_CHARS)
    p.add_argument("--max-digit-frac", type=float, default=C.SENT_MAX_DIGIT_FRAC)
    p.add_argument("--n-process", type=int, default=C.N_PROCESS)
    p.add_argument("--batch-size", type=int, default=C.SPACY_BATCH_SIZE)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_nlp():
    """Blank multilingual spaCy pipeline with just a rule-based sentencizer.

    TODO: `spacy.blank("xx")` + `nlp.add_pipe("sentencizer")`. Blank (not a
    trained model) on purpose - we only need sentence boundaries, not POS/NER,
    and this keeps stage 2 fast and dependency-light.
    """
    raise NotImplementedError


def is_junk(sentence: str, args: argparse.Namespace) -> bool:
    """True if `sentence` should be dropped.

    TODO: length bounds (min/max chars), digit-fraction filter, maybe a
    repeated-punctuation / all-caps heuristic for boilerplate/nav-menu text.
    """
    raise NotImplementedError


def iter_raw_articles():
    """Yield (id, lang, text) dicts from every data/raw/wiki_*.jsonl shard.

    TODO: open each shard, `json.loads` per line, yield records.
    """
    raise NotImplementedError


def process_and_write(args: argparse.Namespace) -> dict:
    """Run the spaCy pipeline over raw articles and write parquet shards.

    TODO:
      - nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process)
      - split each doc into sentences, filter with is_junk()
      - buffer into a DataFrame / pyarrow table, roll a new parquet shard
        at some row-count threshold (mirror ShardWriter's byte-rollover)
      - stop early once args.target_sentences is reached
    Returns a dict of stats for the manifest (counts per language, dropped, etc).
    """
    raise NotImplementedError


def write_manifest(stats: dict, t0: float) -> None:
    shards = sorted(p.name for p in C.SENT_DIR.glob("sent_*.parquet"))
    manifest = {
        "stage": "02_clean_split",
        **stats,
        "shards": shards,
        "n_shards": len(shards),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", MANIFEST)


def main() -> None:
    args = parse_args()
    raw_shards = sorted(C.RAW_DIR.glob("wiki_*.jsonl"))

    LOG.info("target=%s sentences  min_chars=%d  max_chars=%d  max_digit_frac=%.2f  "
             "n_process=%d  batch_size=%d",
             human_int(args.target_sentences), args.min_chars, args.max_chars,
             args.max_digit_frac, args.n_process, args.batch_size)

    if not raw_shards:
        raise SystemExit(f"No input found in {C.RAW_DIR} - run 01_download.py first.")

    if args.dry_run:
        LOG.info("dry-run: would read %d raw shard(s) from %s, write parquet shards to %s",
                  len(raw_shards), C.RAW_DIR, C.SENT_DIR)
        return

    existing = sorted(C.SENT_DIR.glob("sent_*.parquet"))
    if existing and not args.force:
        raise SystemExit(
            f"{C.SENT_DIR} already holds {len(existing)} shard(s). Pass --force to overwrite."
        )
    if args.force and existing:
        LOG.warning("--force: removing %d existing shard(s)", len(existing))
        for f in existing:
            f.unlink()
        MANIFEST.unlink(missing_ok=True)

    t0 = time.time()
    stats = process_and_write(args)
    write_manifest(stats, t0)
    LOG.info("DONE - see %s", MANIFEST)


if __name__ == "__main__":
    main()
