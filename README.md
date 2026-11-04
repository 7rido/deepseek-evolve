# deepseek-evolve

A local DeepSeek-R1 distill (via MLX on Apple Silicon) that improves itself in two ways, with one eval gate deciding what sticks:

- **Loop A — self-fine-tuning.** Run tasks, keep the verified successes, LoRA-train a candidate adapter, promote it only if it beats the champion on held-out tasks.
- **Loop B — innovation.** The model proposes a change to its own harness (prompt, decoding, tools) on a git branch; the same gate decides whether it merges.

Two frontier models collaborate on it through their APIs, each in a role the other cannot take:

- **DeepSeek API (full R1) — the teacher.** Where the 7B student fails a task, the teacher writes a reference solution; if it passes the scorer it joins the training set. This is distillation, and it is what lets the student climb past what it can already do.
- **Claude API — the investigator.** Proposes Loop B experiments with the full report, harness and history in context; DeepSeek R1 critiques each card as a skeptical colleague from a different model family; Claude revises or drops it. Claude also writes the lab notebook (`experiments/NOTEBOOK.md`). Claude's text is research material only and can never enter the training set; `trajectories.py` raises if it does.

The gate, the sandbox, the adapter registry and the eval set are code no model can edit. Without API keys everything still runs with the local model proposing on its own. Design doc: see the Claude doc shared with this repo.

## Requirements

- Apple Silicon Mac, 16 GB+ (32 GB comfortable for the 7B model)
- Python 3.11+
- Docker via OrbStack (for the sandbox; dry runs work without it)
- git

## Setup

```bash
./scripts/setup.sh          # venv, deps, git hook, pin the eval set
source .venv/bin/activate
# pull the base model once (~6.6 GB for the 9B 4-bit)
python -c "from mlx_lm import load; load('mlx-community/Qwen3.5-9B-MLX-4bit')"
```

To shake out the pipeline in minutes, set `model.base` in `config.yaml` to a 1.5B–4B model first.

## API keys

```bash
cp .env.example .env    # add DEEPSEEK_API_KEY and ANTHROPIC_API_KEY; .env is gitignored
```

A nightly cycle is roughly 50 DeepSeek calls (teacher + critic) and 10–20 Claude calls (proposals, revisions, notebook).

## Commands

```bash
evolve eval                 # baseline the champion on tasks/eval_tasks.jsonl
evolve collect              # run seed tasks N times, log scored trajectories
evolve teach                # DeepSeek API solves the tasks the student failed
evolve train                # Loop A: filter -> dataset -> LoRA -> gate -> promote/reject
evolve experiment           # Loop B: up to 5 harness experiments, each gated
evolve cycle                # collect -> teach -> train -> experiment (wire scripts/run_cycle.sh into launchd)
evolve status               # champion, score, generations
evolve rollback             # previous adapter + revert last merged harness change
evolve pin-evals            # after YOU edit tasks/eval_tasks.jsonl
```

Add `--dry-run` to any command to run without MLX or Docker (deterministic stub model). `pytest` runs 17 tests that need neither.

## Layout

```
config.yaml            knobs the loops may NOT change (protected)
harness/               the ONLY directory Loop B may edit
  system_prompt.md
  decoding.yaml
tasks/
  seed_tasks.jsonl     what the agent practices on (Loop A trains on wins here)
  eval_tasks.jsonl     held out; never trained on; hash-pinned (protected)
evolve/
  model.py             ONE shared base in memory; adapter scales toggled for student vs judge
  apis.py              DeepSeek + Anthropic clients (stdlib), tagged by source
  teachers.py          DeepSeek teacher: solve-on-failure -> training data
  notebook.py          lab notebook entries, written by the investigator
  tasks.py             task format + scorers (exact / unit_test / judge)
  sandbox.py           Docker runner, no network, timeouts           (protected)
  trajectories.py      collect -> filter wins -> build SFT dataset
  train.py             mlx_lm.lora -> gate -> promote/reject
  evals.py             the gate: evaluate() + decide()               (protected)
  adapters.py          registry, promote, rollback                   (protected)
  experiments.py       Loop B: Claude proposes, DeepSeek critiques, branch -> gate -> merge (protected)
  harness_loader.py    reads harness/ at call time
  cli.py
scripts/pre-commit     blocks edits outside harness/ on exp/* branches
data/, adapters/, experiments/*.md|jsonl, .env   run-time state and secrets, gitignored
```

## Task format

One JSON object per line. Scorers must be strict: Loop A only trains on `score >= 1.0`.

```json
{"id": "code-001", "prompt": "...", "scorer": "unit_test",
 "tests": "from solution import f\nassert f(2) == 4\n", "tags": ["must_pass"]}
{"id": "arith-001", "prompt": "...", "scorer": "exact", "answer": "391"}
{"id": "explain-001", "prompt": "...", "scorer": "judge", "rubric": "must mention ..."}
```

`must_pass` tasks may never regress; a candidate that fails one is rejected whatever its mean.

## How promotion works

A candidate (adapter or harness branch) first gets a 1-seed cheap pass; only if it clears that is it scored over 3 seeds. It is promoted only if its mean beats the champion's by `gate.promote_margin` (2 points on 0-100) and it has no `must_pass` regressions. Judge-scored tasks always use the frozen base model with no adapter.

After `train.max_generations_from_base` promotions the next Loop A cycle retrains from the base on the full accumulated dataset instead of stacking adapters, which limits drift from training on the model's own outputs.

## Memory on a 16 GB M1 Pro

The judge is the same base model with the LoRA scale set to 0, so only one 4.5 GB copy of the 7B is ever resident. Expect ~32 tok/s decode, ~1 it/s LoRA training (~12 min for the default 600 iterations), and ~1.5 h for a full nightly cycle at the shipped task counts.

## Growing it

The system is bounded by its evals. The highest-leverage work is adding tasks to `tasks/` for the things you actually want the model to be good at, each with a scorer that cannot be gamed. Then `evolve pin-evals` and let the loops run.
