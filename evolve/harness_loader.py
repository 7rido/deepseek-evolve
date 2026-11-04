"""Read the editable harness (harness/) at call time, never at import time.

Loop B edits harness/ on a git branch; the gate re-reads it per run, so a
branch checkout is enough to benchmark a proposal with no process restart.
"""
from __future__ import annotations

import yaml

from .config import CFG, path


def system_prompt() -> str:
    return (path(CFG.paths.harness) / "system_prompt.md").read_text().strip()


def decoding() -> dict:
    p = path(CFG.paths.harness) / "decoding.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def build_messages(user_prompt: str) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt}]
    pre = decoding().get("prefill", "")
    if pre:
        msgs.append({"role": "assistant", "content": pre})
    return msgs
