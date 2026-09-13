#!/usr/bin/env python3
"""Stage 1 - data acquisition.

Streams a size-capped, multilingual slice of `wikimedia/wikipedia` from the
Hugging Face Hub and writes it to local JSONL shards. Design constraints:

  * Never hold more than one article in RAM. We stream with
    `datasets(streaming=True)` and write straight to disk.
  * The network only pulls what we consume (parquet shards fetched lazily),
    so the extracted-text cap effectively caps the download too.
  * Round-robin across languages so the slice is genuinely multilingual
    even if one language's stream stalls.

Output:
  data/raw/wiki_00000.jsonl, wiki_00001.jsonl, ...   one JSON object per line:
      {"id": "en:12", "lang": "en", "title": "...", "url": "...", "text": "..."}
  data/raw/manifest.json    real counts/bytes used by the final README.

Usage:
  python 01_download.py                # config.py defaults (~4 GB, 8 languages)
  python 01_download.py --max-gb 1     # fast iteration slice
  python 01_download.py --languages en fr de
  python 01_download.py --dry-run      # print plan + disk estimate, do nothing
  python 01_download.py --resume       # continue a previous partial run
  python 01_download.py --force        # wipe data/raw and start fresh
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import ShardWriter, fmt_eta, human_bytes, human_int, setup_logging

LOG = setup_logging("01_download", C.LOG_DIR)
MANIFEST = C.RAW_DIR / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--max-gb", type=float, default=C.MAX_DOWNLOAD_GB,
                   help="hard cap on extracted UTF-8 text (GiB)")
    p.add_argument("--languages", nargs="+", default=C.LANGUAGES)
    p.add_argument("--snapshot", default=C.SNAPSHOT)
    p.add_argument("--min-chars", type=int, default=C.MIN_ARTICLE_CHARS)
    p.add_argument("--shard-mb", type=int, default=C.RAW_SHARD_MB)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def preflight_disk(needed_text_bytes: int) -> None:
    need = int(needed_text_bytes * 1.15)  # JSONL key/quote overhead
    free = shutil.disk_usage(C.RAW_DIR).free
    LOG.info("disk: need ~%s of new data, %s free at %s",
             human_bytes(need), human_bytes(free), C.RAW_DIR)
    if free < need:
        raise SystemExit("Not enough free disk for this cap. Lower --max-gb and retry.")
    if free < need * 1.3:
        LOG.warning("Tight disk headroom (<30%% slack). Consider a smaller --max-gb.")


def open_stream(snapshot: str, lang: str, skip: int = 0):
    """Return an iterator over the streaming split, optionally fast-forwarded.

    NOTE: `.skip()` still streams (and may re-download) the skipped shards; it
    just doesn't yield them. Resume trades bandwidth for not re-writing.
    """
    from datasets import load_dataset

    ds = load_dataset(
        C.SOURCE_DATASET,
        name=f"{snapshot}.{lang}",
        split="train",
        streaming=True,
    )
    if skip:
        ds = ds.skip(skip)
    return iter(ds)


def load_prev_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning("existing manifest.json is unreadable; ignoring it")
    return {}


def write_manifest(args, langs, counts, text_bytes, dropped, t0) -> None:
    shards = sorted(p.name for p in C.RAW_DIR.glob("wiki_*.jsonl"))
    manifest = {
        "stage": "01_download",
        "source_dataset": C.SOURCE_DATASET,
        "snapshot": args.snapshot,
        "languages": langs,
        "cap_bytes": int(args.max_gb * 1024**3),
        "text_bytes": int(text_bytes),
        "text_human": human_bytes(text_bytes),
        "articles_total": int(sum(counts.values())),
        "articles_per_language": {l: int(counts.get(l, 0)) for l in langs},
        "short_articles_dropped": int(dropped),
        "min_article_chars": args.min_chars,
        "shards": shards,
        "n_shards": len(shards),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", MANIFEST)


def main() -> None:
    args = parse_args()
    max_bytes = int(args.max_gb * 1024**3)
    langs = list(dict.fromkeys(args.languages))  # dedupe, keep order

    LOG.info("source=%s  snapshot=%s  langs=%s  cap=%s",
             C.SOURCE_DATASET, args.snapshot, ",".join(langs), human_bytes(max_bytes))

    if args.dry_run:
        preflight_disk(max_bytes)
        per = max_bytes / len(langs)
        LOG.info("dry-run: round-robin %d languages (~%s each) into %dMB JSONL shards under %s",
                 len(langs), human_bytes(per), args.shard_mb, C.RAW_DIR)
        return

    existing = sorted(C.RAW_DIR.glob("wiki_*.jsonl"))
    prev_counts: dict[str, int] = {}
    start_shard = 0
    resumed_bytes = 0

    if existing and not (args.resume or args.force):
        raise SystemExit(
            f"{C.RAW_DIR} already holds {len(existing)} shard(s). "
            f"Pass --resume to continue or --force to overwrite."
        )
    if args.force and existing:
        LOG.warning("--force: removing %d existing shard(s)", len(existing))
        for f in existing:
            f.unlink()
        MANIFEST.unlink(missing_ok=True)
        existing = []
    if args.resume and existing:
        prev = load_prev_manifest()
        prev_counts = {k: int(v) for k, v in prev.get("articles_per_language", {}).items()}
        resumed_bytes = int(prev.get("text_bytes", 0))
        start_shard = int(existing[-1].stem.split("_")[-1]) + 1
        LOG.info("resume: %d prior shard(s), %s prior text, per-lang %s",
                 len(existing), human_bytes(resumed_bytes), prev_counts)
        if resumed_bytes >= max_bytes:
            LOG.info("cap already reached by previous run; nothing to do")
            return

    preflight_disk(max_bytes - resumed_bytes)

    LOG.info("opening %d language streams (first article may take ~10-30s each)...", len(langs))
    streams = {l: open_stream(args.snapshot, l, prev_counts.get(l, 0)) for l in langs}

    writer = ShardWriter(C.RAW_DIR, "wiki", args.shard_mb * 1024**2, start_index=start_shard)
    counts: dict[str, int] = {l: prev_counts.get(l, 0) for l in langs}
    seen_first: set[str] = set()
    text_bytes = resumed_bytes
    dropped = 0
    active = list(langs)

    stop = {"flag": False}

    def _sig(signum, _frame):
        LOG.warning("signal %s received - flushing shards and manifest, then exiting", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    t0 = time.time()
    last_log = t0
    last_bytes = text_bytes

    try:
        while active and text_bytes < max_bytes and not stop["flag"]:
            for lang in list(active):
                if text_bytes >= max_bytes or stop["flag"]:
                    break
                try:
                    rec = next(streams[lang])
                except StopIteration:
                    LOG.info("[%s] stream exhausted at %s articles", lang, human_int(counts[lang]))
                    active.remove(lang)
                    continue
                except Exception as exc:  # network hiccup on one language shouldn't kill the run
                    LOG.warning("[%s] stream error (%s); dropping this language", lang, exc)
                    active.remove(lang)
                    continue

                if lang not in seen_first:
                    seen_first.add(lang)
                    LOG.info("[%s] streaming - first article: %r", lang, (rec.get("title") or "")[:60])

                text = (rec.get("text") or "").strip()
                if len(text) < args.min_chars:
                    dropped += 1
                    continue

                writer.write({
                    "id": f'{lang}:{rec.get("id", "")}',
                    "lang": lang,
                    "title": rec.get("title", ""),
                    "url": rec.get("url", ""),
                    "text": text,
                })
                text_bytes += len(text.encode("utf-8"))
                counts[lang] += 1

                now = time.time()
                if now - last_log >= 5.0 or text_bytes >= max_bytes:
                    rate = (text_bytes - last_bytes) / max(1e-6, now - last_log)
                    eta = (max_bytes - text_bytes) / rate if rate > 0 else 0.0
                    LOG.info(
                        "%5.1f%%  %s / %s  | %s articles  | %s/s  | ETA %s  | %s",
                        100 * text_bytes / max_bytes,
                        human_bytes(text_bytes), human_bytes(max_bytes),
                        human_int(sum(counts.values())), human_bytes(rate), fmt_eta(eta),
                        "  ".join(f"{l}={human_int(counts[l])}" for l in langs),
                    )
                    last_log, last_bytes = now, text_bytes
    finally:
        writer.close()
        write_manifest(args, langs, counts, text_bytes, dropped, t0)

    total_shards = len(sorted(C.RAW_DIR.glob("wiki_*.jsonl")))
    LOG.info(
        "DONE  %s text  | %s articles  | %d shard(s)  | %s short articles dropped  | %s elapsed",
        human_bytes(text_bytes), human_int(sum(counts.values())), total_shards,
        human_int(dropped), fmt_eta(time.time() - t0),
    )
    if stop["flag"]:
        LOG.info("stopped early by signal - rerun with --resume to top up to the cap")


if __name__ == "__main__":
    main()
