"""Task bank: load tasks and score responses.

A task is one JSONL line:
  {"id": "...", "prompt": "...", "scorer": "exact|unit_test|judge",
   "answer": "...",          # exact
   "tests": "python src",    # unit_test: code that imports `solution` and asserts
   "rubric": "...",          # judge: what a 1.0 answer looks like
   "tags": ["must_pass", ...]}

Scores are floats in [0, 1]. Loop A keeps trajectories with score >= 1.0,
so a scorer must be strict: partial credit is for reporting, not training.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import sandbox
from .model import LocalModel, strip_think


@dataclass
class Task:
    id: str
    prompt: str
    scorer: str
    answer: str = ""
    tests: str = ""
    rubric: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def must_pass(self) -> bool:
        return "must_pass" in self.tags


def load_tasks(p: Path) -> list[Task]:
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Task(**{k: d[k] for k in Task.__dataclass_fields__ if k in d}))
    return out


# -- scorers ------------------------------------------------------------

def _extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else text


def score_exact(task: Task, response: str) -> float:
    ans = strip_think(response).strip().splitlines()
    return 1.0 if ans and ans[-1].strip().lower() == task.answer.strip().lower() else 0.0


def score_unit_test(task: Task, response: str) -> float:
    code = _extract_code(strip_think(response))
    r = sandbox.run_python(task.tests, files={"solution.py": code})
    return 1.0 if r.ok else 0.0


JUDGE_PROMPT = """You are grading an answer. Reply with ONLY a number from 0 to 10.

Task:
{prompt}

What a perfect answer must contain:
{rubric}

Answer being graded:
{answer}

Score (0-10):"""


def score_judge(task: Task, response: str, judge: LocalModel) -> float:
    msg = JUDGE_PROMPT.format(prompt=task.prompt, rubric=task.rubric,
                              answer=strip_think(response))
    out = strip_think(judge.chat([{"role": "user", "content": msg}],
                                 temperature=0.0, max_tokens=8))
    m = re.search(r"\d+(?:\.\d+)?", out)
    if not m:
        return 0.0
    v = min(10.0, max(0.0, float(m.group())))
    return 1.0 if v >= 9 else v / 10.0


def score(task: Task, response: str, judge: LocalModel | None = None) -> float:
    if task.scorer == "exact":
        return score_exact(task, response)
    if task.scorer == "unit_test":
        return score_unit_test(task, response)
    if task.scorer == "judge":
        if judge is None:
            raise ValueError("judge scorer needs the frozen judge model")
        return score_judge(task, response, judge)
    raise ValueError(f"unknown scorer {task.scorer}")
