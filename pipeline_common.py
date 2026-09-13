"""Shared helpers: logging setup, human-readable formatting, a rolling
JSONL shard writer. Imported by every stage script.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def setup_logging(stage: str, log_dir: Path) -> logging.Logger:
    """Logger that writes to stdout AND logs/<stage>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def human_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ShardWriter:
    """Append-only JSONL writer that rolls to a new file at `shard_bytes`."""

    def __init__(self, out_dir: Path, prefix: str, shard_bytes: int, start_index: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_bytes = int(shard_bytes)
        self.index = int(start_index)
        self._fh = None
        self._cur_bytes = 0
        self.paths: list[Path] = []
        self.records_written = 0

    def _open_new(self) -> None:
        if self._fh:
            self._fh.close()
        path = self.out_dir / f"{self.prefix}_{self.index:05d}.jsonl"
        self._fh = open(path, "w", encoding="utf-8")
        self._cur_bytes = 0
        self.paths.append(path)
        self.index += 1

    def write(self, obj: dict) -> None:
        if self._fh is None or self._cur_bytes >= self.shard_bytes:
            self._open_new()
        line = json.dumps(obj, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._cur_bytes += len(line.encode("utf-8")) + 1
        self.records_written += 1

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
