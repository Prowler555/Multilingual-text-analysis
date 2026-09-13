"""Central configuration for the multilingual text-clustering pipeline.

Every stage script imports from here so that paths and tunables stay
consistent across reruns. Values tagged (TUNE) are expected to change as you
iterate; the clustering grid in particular is meant to be edited.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths - everything lives under the project root; all runtime dirs are
#         gitignored (raw data, embeddings, caches, logs, model weights).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"           # stage 1 -> JSONL shards (wiki_*.jsonl)
SENT_DIR    = DATA_DIR / "sentences"     # stage 2 -> parquet shards of clean sentences
EMB_DIR     = DATA_DIR / "embeddings"    # stage 3 -> float16 memmap + row metadata
REDUCE_DIR  = DATA_DIR / "reduced"       # stage 4/5 -> PCA + UMAP memmaps
RESULTS_DIR = ROOT / "results"           # stage 5/6 -> labels, metrics.json, plots
LOG_DIR     = ROOT / "logs"
CACHE_DIR   = ROOT / "cache"             # Hugging Face datasets + model cache

for _d in (DATA_DIR, RAW_DIR, SENT_DIR, EMB_DIR, REDUCE_DIR, RESULTS_DIR, LOG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep every Hugging Face artefact inside the project (easy to wipe, easy to
# gitignore). Set before `datasets` / `transformers` are imported anywhere.
os.environ.setdefault("HF_HOME", str(CACHE_DIR))

# ---------------------------------------------------------------------------
# Stage 1 - data acquisition
# ---------------------------------------------------------------------------
SOURCE_DATASET    = "wikimedia/wikipedia"   # pre-cleaned, ungated, parquet-backed
SNAPSHOT          = "20231101"              # HF config prefix -> "20231101.<lang>"
LANGUAGES         = ["en", "es", "fr", "de", "ru", "zh", "ja", "ar"]
MAX_DOWNLOAD_GB   = 4.0                     # hard cap on extracted UTF-8 text bytes
RAW_SHARD_MB      = 512                     # roll a new JSONL shard at this size
MIN_ARTICLE_CHARS = 200                     # drop stubs / disambiguation pages

# ---------------------------------------------------------------------------
# Stage 2 - cleaning & sentence splitting            (blank spaCy + sentencizer)
# ---------------------------------------------------------------------------
TARGET_SENTENCES   = 1_200_000   # (TUNE) stop once we have this many clean sentences
SENT_MIN_CHARS     = 30          # too short -> fragment / boilerplate
SENT_MAX_CHARS     = 400         # too long  -> list dump / unsplit table
SENT_MAX_DIGIT_FRAC = 0.30       # drop lines that are mostly numbers
N_PROCESS          = max(1, (os.cpu_count() or 4) - 2)   # spaCy nlp.pipe workers
SPACY_BATCH_SIZE   = 200

# ---------------------------------------------------------------------------
# Stage 3 - embeddings                          (XLM-RoBERTa base, fp16, GPU)
# ---------------------------------------------------------------------------
EMB_MODEL      = "xlm-roberta-base"
EMB_DIM        = 768
EMB_BATCH      = 128        # (TUNE) drop to 64/32 if you see CUDA OOM
EMB_MAX_TOKENS = 64         # short cap - sentence-level text is fine here
EMB_DTYPE      = "float16"  # memmap dtype on disk

# ---------------------------------------------------------------------------
# Stage 4 - dimensionality reduction (PCA)
# ---------------------------------------------------------------------------
PCA_DIMS       = 50
PCA_BATCH      = 20_000     # IncrementalPCA chunk size (keeps RAM bounded)

# ---------------------------------------------------------------------------
# Stage 5 - UMAP + HDBSCAN
# ---------------------------------------------------------------------------
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.0
UMAP_N_COMPONENTS = 10
UMAP_METRIC      = "cosine"
UMAP_LOW_MEMORY  = True

# (TUNE) grid you iterate over in stage 5; each entry -> one HDBSCAN run
HDBSCAN_GRID = [
    {"min_cluster_size": 50,  "min_samples": 10},
    {"min_cluster_size": 100, "min_samples": 15},
    {"min_cluster_size": 200, "min_samples": 25},
    {"min_cluster_size": 500, "min_samples": 50},
]

# ---------------------------------------------------------------------------
# Stage 6 - evaluation
# ---------------------------------------------------------------------------
SILHOUETTE_SAMPLE = 50_000   # O(n^2) metric -> computed on a fixed random sample
RANDOM_SEED       = 42
