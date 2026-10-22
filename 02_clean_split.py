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
import collections
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import human_int, setup_logging

LOG = setup_logging("02_clean_split", C.LOG_DIR)
MANIFEST = C.SENT_DIR / "manifest.json"

# Roll a new parquet shard every this many sentences. Row-count rollover (rather
# than ShardWriter's byte rollover) because parquet is columnar/compressed - a
# fixed row count keeps each shard a predictable chunk for later stages to stream.
SHARD_ROWS = 100_000


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


# CJK sentence terminators. The sentencizer already treats these as boundaries,
# but CJK text has no spaces, so the default tokenizer keeps a whole paragraph as
# one token and no internal boundary is ever set. Inserting a space after each
# terminator (when not already followed by whitespace) makes it its own token, so
# zh/ja actually split into sentences instead of one paragraph-long "sentence".
_CJK_TERM = re.compile(r"([。！？])(?!\s)")

# Any Hiragana, Katakana, or CJK ideograph -> treat the sentence as CJK and use
# the smaller character window (see config.SENT_CJK_*).
_CJK_CHARS = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


def normalize_for_split(text: str) -> str:
    """Light pre-split normalization: space out CJK sentence terminators."""
    return _CJK_TERM.sub(r"\1 ", text)


def build_nlp():
    """Blank multilingual spaCy pipeline with just a rule-based sentencizer.

    Blank (not a trained model) on purpose - we only need sentence boundaries,
    not POS/NER, and this keeps stage 2 fast and dependency-light.
    """
    import spacy  # lazy: keeps --help / --dry-run working without spaCy installed

    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    # Only the rule-based sentencizer runs, so per-character memory is tiny;
    # raise the length cap so long Wikipedia articles don't trip spaCy's
    # default 1M-character guard (which exists to protect the trained pipelines).
    nlp.max_length = max(nlp.max_length, 3_000_000)
    return nlp


def is_junk(sentence: str, args: argparse.Namespace) -> bool:
    """True if `sentence` should be dropped.

    Filters: length bounds, digit-heavy lines (tables / stat dumps), and lines
    with no alphabetic content at all (nav menus, punctuation runs, dates).

    The character bounds are script-aware: CJK sentences are far shorter in
    characters than Latin ones, so they get the smaller SENT_CJK_* window.
    """
    s = sentence.strip()
    n = len(s)
    if _CJK_CHARS.search(s):
        min_chars, max_chars = C.SENT_CJK_MIN_CHARS, C.SENT_CJK_MAX_CHARS
    else:
        min_chars, max_chars = args.min_chars, args.max_chars
    if n < min_chars or n > max_chars:
        return True
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return True
    digit_frac = sum(c.isdigit() for c in non_space) / len(non_space)
    if digit_frac > args.max_digit_frac:
        return True
    if not any(c.isalpha() for c in s):
        return True
    return False


def iter_raw_articles():
    """Yield (id, lang, text) tuples from every data/raw/wiki_*.jsonl shard.

    Streams line-by-line so we never hold a whole shard in memory.
    """
    for shard in sorted(C.RAW_DIR.glob("wiki_*.jsonl")):
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = (rec.get("text") or "").strip()
                if not text:
                    continue
                yield rec.get("id", ""), rec.get("lang", ""), normalize_for_split(text)


class ParquetShardWriter:
    """Buffer sentence rows and roll a new parquet shard every `shard_rows`.

    Mirrors pipeline_common.ShardWriter's rolling behaviour, but columnar:
    one write_table call per shard (id, lang, sentence) instead of per-line JSON.
    """

    def __init__(self, out_dir: Path, prefix: str, shard_rows: int):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self._pq = pq
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_rows = int(shard_rows)
        self.index = 0
        self.records_written = 0
        self.paths: list[Path] = []
        self._ids: list[str] = []
        self._langs: list[str] = []
        self._sents: list[str] = []

    def add(self, id_: str, lang: str, sentence: str) -> None:
        self._ids.append(id_)
        self._langs.append(lang)
        self._sents.append(sentence)
        self.records_written += 1
        if len(self._sents) >= self.shard_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._sents:
            return
        table = self._pa.table(
            {"id": self._ids, "lang": self._langs, "sentence": self._sents}
        )
        path = self.out_dir / f"{self.prefix}_{self.index:05d}.parquet"
        self._pq.write_table(table, path, compression="zstd")
        self.paths.append(path)
        self.index += 1
        self._ids.clear()
        self._langs.clear()
        self._sents.clear()

    def close(self) -> None:
        self._flush()


def process_and_write(args: argparse.Namespace) -> dict:
    """Run the spaCy pipeline over raw articles and write parquet shards.

    Pairs each article's text with its (id, lang) via `nlp.pipe(as_tuples=True)`
    so the metadata rides along through the multiprocessing pool. Stops as soon
    as `args.target_sentences` clean sentences have been written.
    """
    nlp = build_nlp()
    writer = ParquetShardWriter(C.SENT_DIR, "sent", SHARD_ROWS)

    per_lang: collections.Counter = collections.Counter()
    kept = dropped = articles = 0
    target = args.target_sentences
    last_log = time.time()

    # (text, context) pairs -> (doc, context) back out of the worker pool.
    stream = ((text, (id_, lang)) for id_, lang, text in iter_raw_articles())
    docs = nlp.pipe(
        stream,
        as_tuples=True,
        batch_size=args.batch_size,
        n_process=args.n_process,
    )

    reached = False
    for doc, (id_, lang) in docs:
        articles += 1
        for sent in doc.sents:
            s = sent.text.strip()
            if is_junk(s, args):
                dropped += 1
                continue
            writer.add(id_, lang, s)
            per_lang[lang] += 1
            kept += 1
            if kept >= target:
                reached = True
                break

        now = time.time()
        if now - last_log >= 5.0:
            LOG.info(
                "%s kept  | %s dropped  | %s articles  | %d shard(s)",
                human_int(kept), human_int(dropped), human_int(articles), writer.index,
            )
            last_log = now

        if reached:
            LOG.info("reached target of %s sentences", human_int(target))
            break

    writer.close()
    LOG.info(
        "%s sentences kept across %d shard(s)  (%s dropped, %s articles read)",
        human_int(kept), writer.index, human_int(dropped), human_int(articles),
    )

    return {
        "target_sentences": target,
        "sentences_kept": kept,
        "sentences_dropped": dropped,
        "articles_read": articles,
        "target_reached": reached,
        "per_language": dict(sorted(per_lang.items())),
        "shard_rows": SHARD_ROWS,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "max_digit_frac": args.max_digit_frac,
    }


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
