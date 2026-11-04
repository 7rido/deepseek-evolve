"""Adapter registry: lineage, promotion, rollback.

PROTECTED: the model may not edit this file.

adapters/registry.json:
  {"champion": "<name>|null",
   "generations_from_base": 0,
   "history": [{"name", "parent", "score", "promoted_at", "status"}]}
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .config import CFG, path

REG = path(CFG.paths.adapters) / "registry.json"


def _load() -> dict:
    if REG.exists():
        return json.loads(REG.read_text())
    return {"champion": None, "generations_from_base": 0, "history": []}


def _save(reg: dict) -> None:
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=2))


def champion_dir() -> Path:
    return path(CFG.paths.adapters) / "champion"


def champion_score() -> float | None:
    reg = _load()
    for h in reversed(reg["history"]):
        if h["name"] == reg["champion"]:
            return h["score"]
    return None


def generations_from_base() -> int:
    return _load()["generations_from_base"]


def promote(candidate: Path, score: float) -> None:
    reg = _load()
    name = candidate.name
    # Keep the outgoing champion so rollback is a copy, not a retrain.
    if champion_dir().exists() and reg["champion"]:
        keep = path(CFG.paths.adapters) / reg["champion"]
        if not keep.exists():
            shutil.copytree(champion_dir(), keep)
        shutil.rmtree(champion_dir())
    shutil.copytree(candidate, champion_dir())
    reg["history"].append({"name": name, "parent": reg["champion"], "score": score,
                           "promoted_at": time.time(), "status": "champion"})
    reg["champion"] = name
    reg["generations_from_base"] += 1
    _save(reg)


def reject(candidate: Path, score: float, reason: str) -> None:
    reg = _load()
    dest = path(CFG.paths.adapters) / "rejected" / candidate.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists() and not dest.exists():
        shutil.move(str(candidate), dest)
    reg["history"].append({"name": candidate.name, "parent": reg["champion"], "score": score,
                           "promoted_at": None, "status": f"rejected: {reason}"})
    _save(reg)


def rollback() -> str | None:
    """Restore the champion's parent. Returns the restored name or None."""
    reg = _load()
    cur = reg["champion"]
    parent = next((h["parent"] for h in reversed(reg["history"]) if h["name"] == cur), None)
    if champion_dir().exists():
        shutil.rmtree(champion_dir())
    if parent:
        shutil.copytree(path(CFG.paths.adapters) / parent, champion_dir())
    reg["history"].append({"name": cur, "parent": parent, "score": None,
                           "promoted_at": time.time(), "status": "rolled back"})
    reg["champion"] = parent
    reg["generations_from_base"] = max(0, reg["generations_from_base"] - 1)
    _save(reg)
    return parent


def reset_generations() -> None:
    reg = _load()
    reg["generations_from_base"] = 0
    _save(reg)
