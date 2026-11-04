"""The DeepSeek teacher: where the student fails, the full-size model shows
how, and passing solutions join the training set.

This is distillation, which is how the R1 distills were made in the first
place, and it is what lets a 7B climb past what it can already do. Only
``deepseek-api`` text is eligible; the Anthropic model never writes here.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict

from . import apis
from .config import CFG, path
from .harness_loader import build_messages
from .model import LocalModel
from .tasks import Task, load_tasks, score
from .trajectories import _all_trajectories, _store_file

log = logging.getLogger(__name__)

TEACHER_SYSTEM = (
    "You are writing a reference solution that a smaller model will learn from. "
    "Reason carefully, then follow the user's output format exactly. "
    "For numeric answers put only the number on the last line; for code reply "
    "with a single ```python block and nothing after it."
)


def failed_tasks(tasks: list[Task]) -> list[Task]:
    """Tasks where every logged student sample scored below the keep threshold."""
    best: dict[str, float] = defaultdict(float)
    seen: set[str] = set()
    for r in _all_trajectories():
        seen.add(r["task_id"])
        best[r["task_id"]] = max(best[r["task_id"]], r["score"])
    if not CFG.teacher.only_on_student_failure:
        return tasks
    return [t for t in tasks if t.id in seen and best[t.id] < CFG.filter.min_score]


def teach(judge: LocalModel, tasks: list[Task] | None = None) -> int:
    """Ask the teacher for solutions to failed tasks; log passing ones as
    trajectories with source=deepseek-api so Loop A can train on them."""
    if not CFG.teacher.enabled:
        return 0
    if not apis.deepseek_available():
        log.warning("teacher skipped: DEEPSEEK_API_KEY not set")
        return 0
    tasks = tasks or load_tasks(path(CFG.paths.seed_tasks))
    targets = failed_tasks(tasks)
    log.info("teacher: %d of %d seed tasks need a reference solution", len(targets), len(tasks))
    n = 0
    with open(_store_file(), "a") as f:
        for t in targets:
            for attempt in range(CFG.teacher.max_attempts_per_task):
                messages = build_messages(t.prompt)
                api_messages = [{"role": "system", "content": TEACHER_SYSTEM}] + messages[1:]
                try:
                    reply = apis.deepseek_chat(api_messages)
                except apis.APIError as e:
                    log.error("teacher call failed for %s: %s", t.id, e)
                    break
                s = score(t, reply.text, judge)
                rec = {"task_id": t.id, "prompt": t.prompt, "messages": messages,
                       "response": reply.text, "score": s, "adapter": "teacher",
                       "source": reply.source, "teacher_model": reply.model,
                       "ts": time.time(), "sample": attempt}
                f.write(json.dumps(rec) + "\n")
                log.info("teacher %s attempt %d score %.2f (%s tokens)", t.id, attempt, s,
                         reply.usage.get("total_tokens", "?"))
                if s >= CFG.filter.min_score:
                    n += 1
                    break
    log.info("teacher: %d new training examples", n)
    return n
