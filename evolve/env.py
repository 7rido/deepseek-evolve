"""Load .env from the repo root into os.environ (no dependency on python-dotenv)."""
from __future__ import annotations

import os

from .config import ROOT


def load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def deepseek_key() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or None


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None
