"""Trajectory store (Loop A, steps 1-3): collect, filter, build the SFT dataset.

Each trajectory is one JSONL line in data/trajectories/<date>.jsonl:
  {"task_id", "prompt", "messages", "response", "score", "adapter", "ts"}

Append-only. Filtering never deletes; it selects into data/sft/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

from . import apis
from .config import CFG, path
from .harness_loader import build_messages
from .model import LocalModel
from .tasks import Task, load_tasks, score

log = logging.getLogger(__name__)


def _store_file() -> Path:
    d = path(CFG.paths.trajectories)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{time.strftime('%Y-%m-%d')}.jsonl"


def collect(model: LocalModel, judge: LocalModel, tasks: list[Task] | None = None,
            samples: int | None = None, adapter_name: str = "champion") -> int:
    """Run every seed task `samples` times with the champion; log scored runs."""
    tasks = tasks or load_tasks(path(CFG.paths.seed_tasks))
    samples = samples or CFG.collect.samples_per_task
    n = 0
    with open(_store_file(), "a") as f:
        for t in tasks:
            for i in range(samples):
                messages = build_messages(t.prompt)
                resp = model.chat(messages, temperature=CFG.collect.temperature)
                s = score(t, resp, judge)
                rec = {"task_id": t.id, "prompt": t.prompt, "messages": messages,
                       "response": resp, "score": s, "adapter": adapter_name,
                       "source": apis.LOCAL, "ts": time.time(), "sample": i}
                f.write(json.dumps(rec) + "\n")
                n += 1
                log.info("collect %s sample %d score %.2f", t.id, i, s)
    return n


def _all_trajectories() -> list[dict]:
    recs = []
    for p in sorted(path(CFG.paths.trajectories).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _approx_tokens(text: str) -> int:
    return len(text) // 4


TRAINABLE_SOURCES = {apis.LOCAL, apis.DEEPSEEK}


class UntrainableSource(ValueError):
    """Raised if text from a source that may not be trained on reaches the
    dataset builder. Anthropic's terms forbid training on Claude outputs, so
    the investigator's text is research material only."""


def filter_wins(recs: list[dict] | None = None) -> list[dict]:
    """Keep verified successes, dedupe, cap per task. This is the teacher."""
    recs = recs if recs is not None else _all_trajectories()
    seen: set[str] = set()
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        src = r.get("source", apis.LOCAL)
        if src not in TRAINABLE_SOURCES:
            raise UntrainableSource(f"trajectory for {r.get('task_id')} has source {src!r}; "
                                    "only local and deepseek-api text may be trained on")
        if r["score"] < CFG.filter.min_score:
            continue
        if _approx_tokens(r["response"]) > CFG.collect.max_trajectory_tokens:
            continue
        if "Traceback" in r["response"]:
            continue
        h = hashlib.sha1((r["prompt"] + r["response"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        by_task[r["task_id"]].append(r)
    out = []
    for tid, rs in by_task.items():
        rs.sort(key=lambda r: (-r["score"], r.get("source") != apis.DEEPSEEK,
                               len(r["response"])))  # best, teacher before student, then shortest
        out.extend(rs[: CFG.filter.max_per_task])
    log.info("filter: %d trajectories -> %d training examples over %d tasks",
             len(recs), len(out), len(by_task))
    return out


def build_sft_dataset(wins: list[dict] | None = None) -> tuple[int, int]:
    """Write data/sft/{train,valid}.jsonl in mlx_lm chat format.

    Split by task id so the validation set never sees a prompt the model
    trained on; a per-sample split would leak.
    """
    wins = wins if wins is not None else filter_wins()
    if len(wins) < CFG.filter.min_examples_to_train:
        log.warning("only %d examples; need %d to train", len(wins),
                    CFG.filter.min_examples_to_train)
    task_ids = sorted({w["task_id"] for w in wins})
    random.Random(0).shuffle(task_ids)
    n_valid = max(1, int(len(task_ids) * CFG.filter.valid_fraction))
    valid_ids = set(task_ids[:n_valid])
    out = path(CFG.paths.sft)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "valid": 0}
    with open(out / "train.jsonl", "w") as ftr, open(out / "valid.jsonl", "w") as fva:
        for w in wins:
            rec = {"messages": w["messages"] + [{"role": "assistant", "content": w["response"]}]}
            f = fva if w["task_id"] in valid_ids else ftr
            f.write(json.dumps(rec) + "\n")
            counts["valid" if w["task_id"] in valid_ids else "train"] += 1
    return counts["train"], counts["valid"]
