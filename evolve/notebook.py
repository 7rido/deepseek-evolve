"""The lab notebook: experiments/NOTEBOOK.md, one entry per loop run.

Numbers come from the log; the prose comes from the investigator (Claude)
when a key is present, otherwise a plain template. The notebook is the only
place the project explains itself in words, and it is research material,
never training data.
"""
from __future__ import annotations

import json
import logging
import time

from . import adapters, apis
from .config import CFG, path

log = logging.getLogger(__name__)

NOTEBOOK = path(CFG.paths.experiments) / "NOTEBOOK.md"

ENTRY_PROMPT = """You keep the lab notebook for a project that improves a local 7B model through
gated experiments. Write today's entry from the data below. Be specific and brief: what was tried,
what moved and by how much, what you now believe is going on, and the one thing to try next.
Plain prose, no headings, under 200 words. Do not restate the raw records.

Loop: {loop}
Champion score: {champion}
Records:
{records}

Previous entries (most recent last):
{previous}"""


def _tail(n: int = 3) -> str:
    if not NOTEBOOK.exists():
        return "(none)"
    entries = NOTEBOOK.read_text().split("\n## ")
    return "\n## ".join(entries[-n:])[-4000:]


def write_entry(loop: str, records: list[dict]) -> None:
    champ = adapters.champion_score()
    slim = [{k: r.get(k) for k in ("id", "status", "hypothesis", "score", "champion", "reason",
                                   "proposer", "critique") if r.get(k) is not None}
            for r in records]
    body = None
    if apis.anthropic_available():
        try:
            body = apis.claude_chat([{"role": "user", "content": ENTRY_PROMPT.format(
                loop=loop, champion=champ, records=json.dumps(slim, indent=1),
                previous=_tail())}], max_tokens=600).text.strip()
        except apis.APIError as e:
            log.warning("notebook entry fell back to template: %s", e)
    if body is None:
        counts: dict[str, int] = {}
        for r in records:
            counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
        body = (f"{loop}: {len(records)} records ({', '.join(f'{v} {k}' for k, v in counts.items())}). "
                f"Champion score {champ}.")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    if not NOTEBOOK.exists():
        NOTEBOOK.write_text("# Lab notebook\n\nOne entry per loop run. Numbers from the log; "
                            "prose by the investigator model.\n")
    with open(NOTEBOOK, "a") as f:
        f.write(f"\n## {stamp} · {loop}\n\n{body}\n")
