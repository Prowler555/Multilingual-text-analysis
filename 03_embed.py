#!/usr/bin/env python3
"""Stage 3 - sentence embeddings.

Reads the clean-sentence parquet shards from stage 2 and embeds every
sentence with XLM-RoBERTa base (mean/CLS pooling - TBD), batched, in fp16,
on the GPU (RTX 4060 Laptop, 8 GB VRAM). Embeddings are written straight to a
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
from pipeline_common import fmt_eta, human_int, setup_logging

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

    Reads only each shard's footer metadata (num_rows) - never loads the
    sentence column just to count.
    """
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(p).metadata.num_rows for p in shards)


def resolve_device(requested: str | None) -> str:
    """Pick the compute device: explicit --device, else cuda if available."""
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, device: str):
    """Load tokenizer + model on `device`; fp16 on GPU, fp32 on CPU.

    fp16 is the whole point on the 8 GB GPU, but half-precision matmul isn't
    supported for most ops on CPU, so the CPU debug path falls back to fp32.
    Embeddings are cast to fp16 for storage regardless (see `embed_batch`).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    return tokenizer, model


def embed_batch(tokenizer, model, sentences: list[str], max_tokens: int, device: str):
    """Tokenize + forward pass -> mean-pooled fp16 embeddings for one batch.

    Returns an (len(sentences), hidden_size) float16 numpy array on the CPU.
    """
    import torch

    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hidden = model(**enc).last_hidden_state  # (b, seq, dim)

    # Mean-pool over real (non-pad) tokens only.
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (b, seq, 1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / counts

    return pooled.to(torch.float16).cpu().numpy()


def iter_sentences(shards: list[Path]):
    """Yield (id, lang, sentence) across all shards, one shard in RAM at a time."""
    import pandas as pd

    for shard in shards:
        df = pd.read_parquet(shard, columns=["id", "lang", "sentence"])
        for id_, lang, sent in zip(df["id"], df["lang"], df["sentence"]):
            yield id_, lang, sent


def run(args: argparse.Namespace, shards: list[Path], n_total: int) -> dict:
    """Iterate every stage-2 shard, embed in batches, write into the memmap."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    device = resolve_device(args.device)
    LOG.info("loading %s on %s ...", args.model, device)
    tokenizer, model = load_model(args.model, device)

    emb_dim = int(model.config.hidden_size)
    if emb_dim != C.EMB_DIM:
        LOG.warning("model hidden_size=%d != config EMB_DIM=%d; using the model's "
                    "value and recording it in the manifest", emb_dim, C.EMB_DIM)

    emb = np.memmap(MEMMAP_PATH, mode="w+", dtype=np.float16, shape=(n_total, emb_dim))

    ids_buf: list[str] = []
    langs_buf: list[str] = []
    sent_buf: list[str] = []
    per_lang: dict[str, int] = {}

    batch_sents: list[str] = []
    offset = 0
    t0 = time.time()
    last_log = t0

    def flush_batch() -> None:
        nonlocal offset, last_log
        if not batch_sents:
            return
        vecs = embed_batch(tokenizer, model, batch_sents, args.max_tokens, device)
        emb[offset:offset + len(batch_sents)] = vecs
        offset += len(batch_sents)
        batch_sents.clear()
        now = time.time()
        if now - last_log >= 5.0:
            rate = offset / max(1e-6, now - t0)
            eta = (n_total - offset) / rate if rate > 0 else 0.0
            LOG.info("%s / %s embedded  (%.0f sent/s, ETA %s)",
                     human_int(offset), human_int(n_total), rate, fmt_eta(eta))
            last_log = now

    for id_, lang, sent in iter_sentences(shards):
        ids_buf.append(id_)
        langs_buf.append(lang)
        sent_buf.append(sent)
        per_lang[lang] = per_lang.get(lang, 0) + 1
        batch_sents.append(sent)
        if len(batch_sents) >= args.batch_size:
            flush_batch()
    flush_batch()  # trailing partial batch

    emb.flush()
    del emb

    if offset != n_total:
        LOG.warning("embedded %d rows but counted %d - memmap/ids may be misaligned",
                    offset, n_total)

    # Row i of ids.parquet lines up with row i of the memmap.
    pq.write_table(
        pa.table({"id": ids_buf, "lang": langs_buf, "sentence": sent_buf}),
        IDS_PATH,
        compression="zstd",
    )
    LOG.info("wrote %s (%s rows)", IDS_PATH, human_int(len(ids_buf)))

    return {
        "model": args.model,
        "device": device,
        "n_sentences": offset,
        "emb_dim": emb_dim,
        "dtype": C.EMB_DTYPE,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "pooling": "mean",
        "per_language": dict(sorted(per_lang.items())),
    }


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
