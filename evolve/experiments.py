"""Loop B: a harness change is proposed, critiqued, implemented on a branch,
benchmarked by the gate, and merged only if it wins.

PROTECTED: the model may not edit this file.

Who proposes depends on which keys are in .env:
  * Claude (investigator) proposes with the full eval report, harness, and
    experiment history; DeepSeek R1 (critic) reviews the card; Claude
    revises or drops it. Two model families disagreeing is a cheap way to
    kill bad experiments before they cost a benchmark run.
  * With no keys, the local champion proposes on its own (the original loop).

Experiment log: experiments/log.jsonl, one line per card:
  {"id", "hypothesis", "file", "expected", "proposer", "critique", "diff",
   "score", "champion", "status": invalid|crashed|rejected|merged, "ts"}
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path

from . import adapters, apis, evals, notebook
from .config import CFG, PROTECTED_PATHS, path
from .model import LocalModel, champion_model, judge_model, strip_think

log = logging.getLogger(__name__)

LOG = path(CFG.paths.experiments) / "log.jsonl"

PROPOSER_PROMPT = """You are improving the harness of a local coding/reasoning agent.
You may change exactly ONE file under harness/. You may NOT touch anything else.

Current eval report (0-100, per task 0-1):
{report}

Harness files:
{harness}

Past experiments (do not repeat a rejected idea):
{history}

Propose one experiment. Reply with a JSON object and nothing else:
{{"hypothesis": "<one sentence>",
  "file": "harness/<name>",
  "expected": "<which tasks should improve and by roughly how much>",
  "new_content": "<the complete new content of that file>"}}"""


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=path("."), capture_output=True,
                          text=True, check=True).stdout.strip()


def _harness_dump() -> str:
    parts = []
    for p in sorted(path(CFG.paths.harness).iterdir()):
        if p.is_file():
            parts.append(f"=== {p.relative_to(path('.'))} ===\n{p.read_text()}")
    return "\n\n".join(parts)


def _history() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]


def _log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _latest_report() -> str:
    reps = sorted((path(CFG.paths.adapters) / "reports").glob("*.json"))
    return reps[-1].read_text() if reps else "(no report yet; run `evolve eval` first)"


INVESTIGATOR_SYSTEM = """You are the principal investigator on a small research project: making a
7B local model (DeepSeek-R1 distill on an M1 Pro) measurably better at a fixed task set by changing
only its harness (prompt, decoding, few-shot files, tool descriptions). You cannot change its weights,
the eval set, or the gate. Every proposal is a falsifiable claim: name the tasks you expect to move and
by how much. Prefer one clean change over several. Do not repeat rejected ideas unless you can say
what is different this time. Reply with the JSON object requested and nothing else."""

CRITIC_PROMPT = """Another model proposed this experiment on a local LLM's harness:

{card}

Current eval report:
{report}

Review it as a skeptical colleague. Is the expected effect plausible given the per-task scores?
Could the change hurt a task it does not mention? Is there a cheaper variant that tests the same idea?
Reply with ONLY a JSON object:
{{"verdict": "proceed" | "revise" | "drop",
  "reasons": "<two or three sentences>",
  "suggestion": "<a concrete change to the card, or empty>"}}"""

REVISE_PROMPT = """Your card:
{card}

A reviewer from a different model family replied:
{critique}

Either revise the card (reply with the full JSON card again) or drop it (reply with {{"drop": true}}).
Reply with JSON and nothing else."""


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", strip_think(text), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _valid_card(card: dict | None) -> bool:
    return bool(card) and all(k in card for k in ("hypothesis", "file", "expected", "new_content"))


def _proposer_context() -> str:
    hist = [{k: h[k] for k in ("hypothesis", "file", "status", "score", "critique") if k in h}
            for h in _history()[-20:]]
    return PROPOSER_PROMPT.format(report=_latest_report(), harness=_harness_dump(),
                                  history=json.dumps(hist, indent=1))


def propose_local(model: LocalModel) -> dict | None:
    out = model.chat([{"role": "user", "content": _proposer_context()}], temperature=0.8)
    card = _parse_json(out)
    return card if _valid_card(card) else None


def propose_collaborative() -> tuple[dict | None, dict]:
    """Claude proposes, DeepSeek critiques, Claude revises or drops.
    Returns (card or None, exchange) where exchange is logged whatever happens."""
    exchange: dict = {"proposer": apis.ANTHROPIC, "critic": apis.DEEPSEEK}
    ctx = _proposer_context()
    try:
        first = apis.claude_chat([{"role": "user", "content": ctx}], system=INVESTIGATOR_SYSTEM)
    except apis.APIError as e:
        exchange["error"] = f"proposer: {e}"
        return None, exchange
    card = _parse_json(first.text)
    exchange["proposed"] = card
    if not _valid_card(card):
        exchange["error"] = "proposer returned no valid card"
        return None, exchange

    if apis.deepseek_available():
        try:
            crit = apis.deepseek_chat(
                [{"role": "user", "content": CRITIC_PROMPT.format(
                    card=json.dumps({k: card[k] for k in ("hypothesis", "file", "expected")}, indent=1)
                         + "\n\nnew_content:\n" + card["new_content"],
                    report=_latest_report())}],
                model=CFG.apis.deepseek_critic_model)
            critique = _parse_json(crit.text) or {"verdict": "proceed",
                                                   "reasons": strip_think(crit.text)[:500]}
        except apis.APIError as e:
            critique = {"verdict": "proceed", "reasons": f"critic unavailable: {e}"}
        exchange["critique"] = critique
        if critique.get("verdict") in ("revise", "drop"):
            try:
                second = apis.claude_chat(
                    [{"role": "user", "content": ctx},
                     {"role": "assistant", "content": first.text},
                     {"role": "user", "content": REVISE_PROMPT.format(
                         card=json.dumps(card, indent=1), critique=json.dumps(critique, indent=1))}],
                    system=INVESTIGATOR_SYSTEM)
                revised = _parse_json(second.text)
            except apis.APIError as e:
                revised = None
                exchange["error"] = f"revise: {e}"
            exchange["revised"] = revised
            if not revised or revised.get("drop"):
                exchange["outcome"] = "dropped after critique"
                return None, exchange
            if _valid_card(revised):
                card = revised
    exchange["outcome"] = "proceed"
    return card, exchange


def propose(model: LocalModel) -> tuple[dict | None, dict]:
    if apis.anthropic_available():
        return propose_collaborative()
    return propose_local(model), {"proposer": apis.LOCAL}


def _is_allowed(file: str) -> bool:
    f = Path(file)
    inside = f.parts and f.parts[0] == CFG.paths.harness and ".." not in f.parts
    return bool(inside) and file not in PROTECTED_PATHS


def run_one(card: dict, dry_run: bool = False, exchange: dict | None = None) -> dict:
    exp_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000:03d}"
    rec = {"id": exp_id, "ts": time.time(), **{k: card[k] for k in ("hypothesis", "file", "expected")}}
    if exchange:
        rec["proposer"] = exchange.get("proposer")
        if "critique" in exchange:
            rec["critique"] = exchange["critique"]
    if not _is_allowed(card["file"]):
        rec["status"] = "invalid"
        rec["reason"] = "file outside harness/"
        _log(rec)
        return rec

    if _git("status", "--porcelain", "--untracked-files=no"):
        rec["status"] = "invalid"
        rec["reason"] = "working tree has uncommitted changes; commit or stash first"
        _log(rec)
        return rec

    branch = f"{CFG.experiments.branch_prefix}{exp_id}"
    base = _git("rev-parse", "--abbrev-ref", "HEAD")
    _git("checkout", "-b", branch)
    try:
        target = path(card["file"])
        target.write_text(card["new_content"])
        _git("add", card["file"])
        _git("commit", "-m", f"exp {exp_id}: {card['hypothesis'][:72]}")
        rec["diff"] = _git("diff", f"{base}..{branch}")

        # Belt and braces: the hook should have caught this, check anyway.
        touched = _git("diff", "--name-only", f"{base}..{branch}").splitlines()
        if any(t in PROTECTED_PATHS or not _is_allowed(t) for t in touched):
            raise PermissionError(f"branch touched protected files: {touched}")

        judge = judge_model(dry_run)
        model = champion_model(dry_run)
        champ = adapters.champion_score()
        ok, why, rep = evals.gate(model, judge, f"exp-{exp_id}", champ)
        rec["score"], rec["champion"] = rep.mean, champ
        rec["reason"] = why
        if ok:
            _git("checkout", base)
            _git("merge", "--no-ff", branch, "-m", f"merge exp {exp_id}: {why}")
            _git("tag", f"harness-{exp_id}")
            rec["status"] = "merged"
        else:
            _git("checkout", base)
            rec["status"] = "rejected"
    except Exception as e:  # noqa: BLE001 - any failure is a crashed experiment
        log.exception("experiment %s crashed", exp_id)
        rec["status"] = "crashed"
        rec["reason"] = str(e)
        _git("checkout", "-f", base)
    finally:
        _git("branch", "-D", branch) if rec.get("status") != "merged" else None
    _log(rec)
    return rec


def cycle(dry_run: bool = False) -> list[dict]:
    """Up to max_per_cycle experiments or wall_clock_minutes, whichever first."""
    deadline = time.time() + 60 * CFG.experiments.wall_clock_minutes
    proposer = champion_model(dry_run)
    results = []
    for i in range(CFG.experiments.max_per_cycle):
        if time.time() > deadline:
            log.info("wall clock budget hit after %d experiments", i)
            break
        card, exchange = propose(proposer)
        if card is None:
            rec = {"id": time.strftime("%Y%m%d-%H%M%S"), "status": "invalid",
                   "reason": exchange.get("error") or exchange.get("outcome")
                   or "proposer returned no valid card", "ts": time.time(),
                   "proposer": exchange.get("proposer"), "critique": exchange.get("critique"),
                   "hypothesis": (exchange.get("proposed") or {}).get("hypothesis")}
            _log(rec)
            results.append(rec)
            continue
        rec = run_one(card, dry_run=dry_run, exchange=exchange)
        results.append(rec)
        log.info("experiment %s -> %s (%s)", rec["id"], rec["status"], rec.get("reason"))
    notebook.write_entry("Loop B", results)
    return results
