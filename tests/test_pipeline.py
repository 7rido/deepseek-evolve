"""Runs anywhere (no MLX, no Docker): exercises filtering, dataset split, the
gate's decision logic, the adapter registry and the protected-path check."""
import json
from pathlib import Path

import pytest

from evolve import adapters, evals, experiments, tasks, trajectories
from evolve.config import CFG, path
from evolve.evals import Report
from evolve.model import DryRunModel, strip_think


def test_strip_think():
    assert strip_think("<think>a\nb</think>\n42") == "42"
    assert strip_think("42") == "42"


def test_exact_scorer():
    t = tasks.Task(id="x", prompt="2+2", scorer="exact", answer="4")
    assert tasks.score(t, "<think>..</think>\n4") == 1.0
    assert tasks.score(t, "<think>..</think>\n5") == 0.0


def test_unit_test_scorer_subprocess():
    t = tasks.Task(id="x", prompt="", scorer="unit_test",
                   tests="from solution import f\nassert f(2) == 4\n")
    assert tasks.score(t, "```python\ndef f(x):\n    return x * 2\n```") == 1.0
    assert tasks.score(t, "```python\ndef f(x):\n    return x\n```") == 0.0


def test_filter_and_split(tmp_path, monkeypatch):
    recs = []
    for tid in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"):
        for i in range(4):
            recs.append({"task_id": tid, "prompt": f"p{tid}", "messages": [],
                         "response": f"r{tid}{i}" * (i + 1), "score": 1.0 if i < 3 else 0.0})
    wins = trajectories.filter_wins(recs)
    assert len(wins) == 10 * CFG.filter.max_per_task
    assert all(w["score"] >= 1.0 for w in wins)
    monkeypatch.setattr(CFG.paths, "sft", str(tmp_path))
    n_tr, n_va = trajectories.build_sft_dataset(wins)
    assert n_tr + n_va == len(wins)
    valid_ids = {json.loads(l)["messages"][-1]["content"][1]
                 for l in (tmp_path / "valid.jsonl").read_text().splitlines()}
    train_ids = {json.loads(l)["messages"][-1]["content"][1]
                 for l in (tmp_path / "train.jsonl").read_text().splitlines()}
    assert not (valid_ids & train_ids), "split leaked task ids"


def _rep(mean, failures=()):
    return Report("x", mean, 0.0, {}, list(failures), 1, "h")


def test_gate_decision():
    assert evals.decide(_rep(50), None)[0]
    assert not evals.decide(_rep(51), 50)[0]            # below margin
    assert evals.decide(_rep(50 + CFG.gate.promote_margin), 50)[0]
    assert not evals.decide(_rep(90, ["ev-1"]), 50)[0]  # must_pass regression beats any delta


def test_registry_promote_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(CFG.paths, "adapters", str(tmp_path))
    monkeypatch.setattr(adapters, "REG", tmp_path / "registry.json")
    c1 = tmp_path / "candidate-1"; c1.mkdir(); (c1 / "adapters.safetensors").write_bytes(b"1")
    c2 = tmp_path / "candidate-2"; c2.mkdir(); (c2 / "adapters.safetensors").write_bytes(b"2")
    adapters.promote(c1, 60.0)
    adapters.promote(c2, 65.0)
    assert adapters.champion_score() == 65.0
    assert adapters.generations_from_base() == 2
    assert (tmp_path / "champion" / "adapters.safetensors").read_bytes() == b"2"
    assert adapters.rollback() == "candidate-1"
    assert (tmp_path / "champion" / "adapters.safetensors").read_bytes() == b"1"


@pytest.mark.parametrize("f,ok", [
    ("harness/system_prompt.md", True),
    ("harness/new_tool.md", True),
    ("evolve/evals.py", False),
    ("config.yaml", False),
    ("harness/../evolve/evals.py", False),
])
def test_protected_paths(f, ok):
    assert experiments._is_allowed(f) == ok


def test_dry_run_model_and_tasks_load():
    seed = tasks.load_tasks(path(CFG.paths.seed_tasks))
    ev = tasks.load_tasks(path(CFG.paths.eval_tasks))
    assert seed and ev
    assert not ({t.id for t in seed} & {t.id for t in ev}), "eval tasks must be held out"
    m = DryRunModel()
    assert strip_think(m.chat([{"role": "user", "content": "What is 2+2?"}])) == "4"


def test_anthropic_text_can_never_be_trained_on():
    recs = [{"task_id": "a", "prompt": "p", "messages": [], "response": "r",
             "score": 1.0, "source": "anthropic-api"}]
    with pytest.raises(trajectories.UntrainableSource):
        trajectories.filter_wins(recs)


def test_teacher_solutions_preferred_over_student_on_ties():
    recs = [{"task_id": "a", "prompt": "p", "messages": [], "response": "student answer",
             "score": 1.0, "source": "local"},
            {"task_id": "a", "prompt": "p", "messages": [], "response": "teacher answer that is longer",
             "score": 1.0, "source": "deepseek-api"}]
    wins = trajectories.filter_wins(recs)
    assert wins[0]["source"] == "deepseek-api"


def test_two_stage_gate_short_circuits(monkeypatch):
    calls = []
    def fake_eval(model, judge, label, tasks=None, seeds=None):
        calls.append(seeds)
        return _rep(10.0)  # always loses
    monkeypatch.setattr(evals, "evaluate", fake_eval)
    ok, why, _ = evals.gate(None, None, "x", champion_mean=50.0)
    assert not ok and calls == [1] and why.startswith("stage 1")


def test_two_stage_gate_confirms(monkeypatch):
    calls = []
    def fake_eval(model, judge, label, tasks=None, seeds=None):
        calls.append(seeds)
        return _rep(80.0)
    monkeypatch.setattr(evals, "evaluate", fake_eval)
    ok, why, _ = evals.gate(None, None, "x", champion_mean=50.0)
    assert ok and calls == [1, CFG.gate.seeds] and why.startswith("stage 2")


def test_apis_fail_cleanly_without_keys(monkeypatch):
    from evolve import apis
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not apis.deepseek_available() and not apis.anthropic_available()
    with pytest.raises(apis.APIError):
        apis.deepseek_chat([{"role": "user", "content": "x"}])
    with pytest.raises(apis.APIError):
        apis.claude_chat([{"role": "user", "content": "x"}])
