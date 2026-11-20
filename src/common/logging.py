"""Console + file logging and a tiny CSV metric logger."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import ensure_dir, resolve_path


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create (or reset) a logger with a console handler and optional file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Reconfiguring the same logger should not duplicate messages.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        p = resolve_path(log_file)
        assert p is not None
        p.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """Append one row per evaluation to a CSV file.

    The header is inferred from the first row; later rows may omit keys
    (they are written empty).  Useful for ``checkpoints/.../metrics.csv``.
    """

    def __init__(self, path: str):
        self.path = resolve_path(path)
        assert self.path is not None
        ensure_dir(str(self.path.parent))
        self.fieldnames: Optional[List[str]] = None
        self._rows: List[Dict[str, Any]] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                self.fieldnames = list(reader.fieldnames or [])
                self._rows = [dict(row) for row in reader]

    def log(self, row: Dict[str, Any]) -> None:
        self._rows.append(dict(row))
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        else:
            for key in row:
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
        self._flush()

    def _flush(self) -> None:
        assert self.path is not None and self.fieldnames is not None
        with open(self.path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


def log_config(logger: logging.Logger, title: str, config: Dict[str, Any]) -> None:
    import json

    logger.info("%s:\n%s", title, json.dumps(config, ensure_ascii=False, indent=2, default=str))
