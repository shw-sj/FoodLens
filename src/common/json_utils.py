"""JSON / JSONL helpers and numpy/torch-safe serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import resolve_path


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/torch/path/set values into JSON-ready ones."""
    try:  # torch is optional for pure metrics tests.
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def dumps(data: Any, indent: int = 2) -> str:
    return json.dumps(to_jsonable(data), ensure_ascii=False, indent=indent)


def dump(data: Any, path: str, indent: int = 2) -> None:
    p = resolve_path(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(dumps(data, indent=indent))
        f.write("\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = resolve_path(path)
    assert p is not None
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    p = resolve_path(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False))
            f.write("\n")
