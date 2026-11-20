"""Small I/O helpers shared by every task.

All *relative* paths in YAML configs and on the command line are resolved
against the repository root (the parent of ``src/``), not against the current
working directory.  This makes ``python -m src.task1_caption.train`` behave
the same whether it is launched from the project root or from a cluster job
directory.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Optional[str]) -> Optional[Path]:
    """Resolve a path against the project root; absolute paths are untouched."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def ensure_dir(path: Optional[str]) -> Path:
    p = resolve_path(path) if path is not None else PROJECT_ROOT
    assert p is not None
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str) -> Dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def save_yaml(data: Dict[str, Any], path: str) -> None:
    p = resolve_path(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_json(path: str, default: Any = None) -> Any:
    p = resolve_path(path)
    if p is None or not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    p = resolve_path(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        f.write("\n")


def load_ids(path: Optional[str]) -> List[str]:
    """Read a manifest ID list: one ID per line, ``#`` comments allowed."""
    if path is None:
        return []
    p = resolve_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"ID list not found: {path} (resolved to {p})")
    ids: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                ids.append(value)
    return ids


def save_ids(ids: Iterable[str], path: str) -> None:
    p = resolve_path(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for item in ids:
            f.write(f"{item}\n")


def read_csv(path: str) -> List[Dict[str, str]]:
    p = resolve_path(path)
    assert p is not None
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(
    rows: Sequence[Dict[str, Any]],
    path: str,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    p = resolve_path(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def apply_overrides(config: Dict[str, Any], overrides: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Apply ``--set a.b=value`` overrides to a config dict.

    Values are parsed with YAML, so ``false``, ``3``, ``1e-4`` and ``null``
    get the expected Python types.  Returns a deep copy.
    """
    cfg = copy.deepcopy(config)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"bad override {item!r}, expected KEY=VALUE")
        value = yaml.safe_load(raw)
        node: Dict[str, Any] = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (used by prepare/train/eval wrappers)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def require_keys(cfg: Dict[str, Any], keys: Sequence[str], context: str = "config") -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"{context} is missing required keys: {', '.join(missing)}")


def to_posix(path: Path) -> str:
    return path.as_posix()
