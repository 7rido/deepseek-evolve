"""The eval gate. PROTECTED: the model may not edit this file.

``evaluate`` scores a model on the held-out set across N seeds and returns a
Report. ``decide`` compares a candidate Report against the champion's cached
score and says promote / reject with a reason. Both loops call exactly these
two functions; nothing else decides what becomes the champion.
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CFG, file_hash, path
from .harness_loader import build_messages, decoding
from .model import LocalModel
from .tasks import Task, load_tasks, score

log = logging.getLogger(__name__)

EVAL_HASH_FILE = path(CFG.paths.adapters) / "eval_tasks.sha256"


@dataclass
class Report:
    label: str
    mean: float                      # 0-100
    stdev: float
    per_task: dict[str, float]       # task_id -> mean score 0-1
    must_pass_failures: list[str]
    seeds: int
    eval_hash: str
    ts: float = field(default_factory=time.time)

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(asdict(self), indent=2))


def check_eval_set_untouched() -> str:
    """Refuse to run if tasks/eval_tasks.jsonl changed since it was pinned.

    Pin it yourself with ``evolve pin-evals`` whenever you curate the set.
    """
    h = file_hash(CFG.paths.eval_tasks)
    if EVAL_HASH_FILE.exists() and EVAL_HASH_FILE.read_text().strip() != h:
        raise RuntimeError("eval_tasks.jsonl changed since it was pinned; run `evolve pin-evals` "
                           "yourself if the change was intentional")
    return h


def pin_eval_set() -> None:
    EVAL_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_HASH_FILE.write_text(file_hash(CFG.paths.eval_tasks))


def evaluate(model: LocalModel, judge: LocalModel, label: str,
             tasks: list[Task] | None = None, seeds: int | None = None) -> Report:
    h = check_eval_set_untouched()
    tasks = tasks or load_tasks(path(CFG.paths.eval_tasks))
    seeds = seeds or CFG.gate.seeds
    temp = float(decoding().get("temperature", 0.2))
    per_task: dict[str, list[float]] = {t.id: [] for t in tasks}
    run_means: list[float] = []
    for seed in range(seeds):
        total = 0.0
        for t in tasks:
            resp = model.chat(build_messages(t.prompt), temperature=temp if seed else 0.0)
            s = score(t, resp, judge)
            per_task[t.id].append(s)
            total += s
        run_means.append(100.0 * total / len(tasks))
        log.info("eval %s seed %d: %.1f", label, seed, run_means[-1])
    pt = {k: statistics.mean(v) for k, v in per_task.items()}
    failures = [t.id for t in tasks if t.must_pass and pt[t.id] < 1.0]
    rep = Report(label=label, mean=statistics.mean(run_means),
                 stdev=statistics.stdev(run_means) if len(run_means) > 1 else 0.0,
                 per_task=pt, must_pass_failures=failures, seeds=seeds, eval_hash=h)
    rep.save(path(CFG.paths.adapters) / "reports" / f"{label}.json")
    return rep


def gate(model: LocalModel, judge: LocalModel, label: str,
         champion_mean: float | None) -> tuple[bool, str, Report]:
    """Two-stage gate: a 1-seed cheap pass, then the full N-seed confirmation
    only for candidates that clear it. Cuts the cost of a losing candidate
    by ~3x, which is what lets the eval set grow to 50+ tasks."""
    cheap = evaluate(model, judge, label=f"{label}-stage1", seeds=1)
    ok, why = decide(cheap, champion_mean)
    if not ok:
        return False, f"stage 1: {why}", cheap
    if CFG.gate.seeds <= 1:
        return True, why, cheap
    full = evaluate(model, judge, label=label, seeds=CFG.gate.seeds)
    ok, why = decide(full, champion_mean)
    return ok, f"stage 2: {why}", full


def decide(candidate: Report, champion_mean: float | None) -> tuple[bool, str]:
    if candidate.must_pass_failures:
        return False, f"must_pass regressions: {candidate.must_pass_failures}"
    if champion_mean is None:
        return True, "no champion yet"
    delta = candidate.mean - champion_mean
    if delta < CFG.gate.promote_margin:
        return False, f"delta {delta:+.1f} below margin {CFG.gate.promote_margin}"
    return True, f"delta {delta:+.1f}"
