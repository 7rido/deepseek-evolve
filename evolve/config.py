"""Load config.yaml once and expose it as nested attribute access.

The loops never write this file; the pre-commit hook rejects any diff to it
from an experiment branch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# Files the model must never modify. Checked by the pre-commit hook and by
# experiments.py before a branch is benchmarked.
PROTECTED_PATHS = [
    "config.yaml",
    "evolve/evals.py",
    "evolve/sandbox.py",
    "evolve/experiments.py",
    "evolve/adapters.py",
    "tasks/eval_tasks.jsonl",
    "scripts/pre-commit",
]


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load(path: Path = CONFIG_PATH) -> SimpleNamespace:
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))


def path(rel: str) -> Path:
    """Resolve a path relative to the repo root."""
    return ROOT / rel


def file_hash(rel: str) -> str:
    """SHA-256 of a repo file; used to prove the eval set was not touched."""
    return hashlib.sha256(path(rel).read_bytes()).hexdigest()


CFG = load()
