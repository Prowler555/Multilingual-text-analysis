#!/usr/bin/env python3
"""Stage 3 - sentence embeddings.

Reads the clean-sentence parquet shards from stage 2 and embeds every
sentence with XLM-RoBERTa base (mean/CLS pooling - TBD), batched, in fp16,
on the GPU (RTX 4050, 6 GB VRAM). Embeddings are written straight to a
pre-sized float16 memmap on disk so the full matrix never has to fit in RAM
or VRAM at once.

Design constraints:
  * Batched inference with a token cap (`EMB_MAX_TOKENS`) - sentence-level
    text is short, so this keeps batches small and predictable in VRAM.
  * Write into a memmap (`emb.f16.memmap`), not a growing Python list, so a
    crash partway through only costs the current batch, not the whole run.
  * Row i of the memmap must line up with row i of `data/embeddings/ids.parquet`
    (id, lang, sentence) so later stages can trace a cluster back to text.

Output:
  data/embeddings/emb.f16.memmap   shape (n_sentences, EMB_DIM), dtype float16
  data/embeddings/ids.parquet      row metadata aligned to the memmap
  data/embeddings/manifest.json

Usage:
  python 03_embed.py                  # config.py defaults
  python 03_embed.py --device cpu     # force CPU (slow - debugging only)
  python 03_embed.py --dry-run        # print plan, do nothing
  python 03_embed.py --force          # wipe data/embeddings and start fresh
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config as C
from pipeline_common import human_int, setup_logging

LOG = setup_logging("03_embed", C.LOG_DIR)
MANIFEST = C.EMB_DIR / "manifest.json"
MEMMAP_PATH = C.EMB_DIR / "emb.f16.memmap"
IDS_PATH = C.EMB_DIR / "ids.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=C.EMB_MODEL)
    p.add_argument("--batch-size", type=int, default=C.EMB_BATCH)
    p.add_argument("--max-tokens", type=int, default=C.EMB_MAX_TOKENS)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto-detect)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def count_sentences(shards: list[Path]) -> int:
    """Total row count across all stage-2 parquet shards.

    TODO: cheapest way is `pyarrow.parquet.ParquetFile(p).metadata.num_rows`
    per shard, summed - avoids loading actual sentence text just to count.
    """
    raise NotImplementedError


def load_model(model_name: str, device: str):
    """Load tokenizer + model in fp16 on `device`.

    TODO: `AutoTokenizer.from_pretrained`, `AutoModel.from_pretrained(...,
    torch_dtype=torch.float16).to(device).eval()`.
    """
    raise NotImplementedError


def embed_batch(tokenizer, model, sentences: list[str], max_tokens: int, device: str):
    """Tokenize + forward pass -> pooled embeddings for one batch.

    TODO: tokenize with padding/truncation to max_tokens, run under
    `torch.no_grad()`, mean-pool the last hidden state over the attention
    mask, cast to float16, move to CPU/numpy.
    """
    raise NotImplementedError


def run(args: argparse.Namespace, shards: list[Path], n_total: int) -> dict:
    """Iterate every stage-2 shard, embed in batches, write into the memmap.

    TODO:
      - open MEMMAP_PATH as np.memmap(mode="w+", shape=(n_total, EMB_DIM), dtype=float16)
      - stream sentences shard by shard (pandas.read_parquet per shard, not all at once)
      - batch into groups of args.batch_size, call embed_batch(), write rows at the
        right offset, flush the memmap
      - append (id, lang, sentence) rows to an ids buffer -> write ids.parquet at the end
    Returns a dict of stats for the manifest.
    """
    raise NotImplementedError


def write_manifest(stats: dict, t0: float) -> None:
    manifest = {
        "stage": "03_embed",
        **stats,
        "memmap_path": str(MEMMAP_PATH),
        "ids_path": str(IDS_PATH),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", MANIFEST)


def main() -> None:
    args = parse_args()
    shards = sorted(C.SENT_DIR.glob("sent_*.parquet"))

    LOG.info("model=%s  batch_size=%d  max_tokens=%d  device=%s",
              args.model, args.batch_size, args.max_tokens, args.device or "auto")

    if not shards:
        raise SystemExit(f"No input found in {C.SENT_DIR} - run 02_clean_split.py first.")

    if args.dry_run:
        LOG.info("dry-run: would embed sentences from %d shard(s) in %s -> %s (dim=%d, fp16)",
                  len(shards), C.SENT_DIR, MEMMAP_PATH, C.EMB_DIM)
        return

    if MEMMAP_PATH.exists() and not args.force:
        raise SystemExit(f"{MEMMAP_PATH} already exists. Pass --force to overwrite.")
    if args.force and MEMMAP_PATH.exists():
        LOG.warning("--force: removing existing embeddings")
        MEMMAP_PATH.unlink()
        IDS_PATH.unlink(missing_ok=True)
        MANIFEST.unlink(missing_ok=True)

    n_total = count_sentences(shards)
    LOG.info("%s sentences to embed across %d shard(s)", human_int(n_total), len(shards))

    t0 = time.time()
    stats = run(args, shards, n_total)
    write_manifest(stats, t0)
    LOG.info("DONE - see %s", MANIFEST)


if __name__ == "__main__":
    main()
