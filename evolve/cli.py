"""`evolve` command line.

  evolve collect [--samples N] [--dry-run]   run seed tasks, log trajectories
  evolve teach   [--dry-run]                 DeepSeek API solves what the student failed
  evolve train   [--dry-run]                 Loop A: build dataset, LoRA, gate, promote
  evolve experiment [--dry-run]              Loop B: propose/implement/benchmark
  evolve eval    [--label L] [--dry-run]     score the champion on the held-out set
  evolve cycle   [--dry-run]                 collect -> train -> experiment
  evolve pin-evals                           record the eval set hash (after curating it)
  evolve rollback                            restore previous adapter + revert harness tag
  evolve status                              champion, generations, last reports
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess

from . import adapters, evals, experiments, teachers, train, trajectories
from .config import CFG, path
from .model import champion_model, judge_model


def cmd_collect(a):
    n = trajectories.collect(champion_model(a.dry_run), judge_model(a.dry_run),
                             samples=a.samples)
    print(f"logged {n} trajectories")


def cmd_teach(a):
    n = teachers.teach(judge_model(a.dry_run))
    print(f"teacher added {n} training examples")


def cmd_train(a):
    print("promoted" if train.cycle(dry_run=a.dry_run) else "no promotion")


def cmd_experiment(a):
    for r in experiments.cycle(dry_run=a.dry_run):
        print(f"{r['id']}: {r['status']}  {r.get('reason', '')}")


def cmd_eval(a):
    rep = evals.evaluate(champion_model(a.dry_run), judge_model(a.dry_run),
                         label=a.label or "champion")
    print(json.dumps({"mean": rep.mean, "stdev": rep.stdev,
                      "must_pass_failures": rep.must_pass_failures}, indent=2))


def cmd_cycle(a):
    cmd_collect(a)
    cmd_teach(a)
    cmd_train(a)
    cmd_experiment(a)


def cmd_pin(a):
    evals.pin_eval_set()
    print("eval set pinned")


def cmd_rollback(a):
    restored = adapters.rollback()
    print(f"adapter: restored {restored or 'base (no adapter)'}")
    tags = subprocess.run(["git", "tag", "-l", "harness-*", "--sort=-creatordate"],
                          cwd=path("."), capture_output=True, text=True).stdout.split()
    if tags:
        subprocess.run(["git", "revert", "--no-edit", "-m", "1", tags[0]], cwd=path("."), check=True)
        print(f"harness: reverted {tags[0]}")


def cmd_status(a):
    reg = adapters._load()
    print(json.dumps({"champion": reg["champion"], "score": adapters.champion_score(),
                      "generations_from_base": reg["generations_from_base"],
                      "history_len": len(reg["history"])}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="evolve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="no MLX, deterministic stub model")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("collect"); s.add_argument("--samples", type=int); s.set_defaults(fn=cmd_collect)
    sub.add_parser("teach").set_defaults(fn=cmd_teach)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("experiment").set_defaults(fn=cmd_experiment)
    s = sub.add_parser("eval"); s.add_argument("--label"); s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("cycle"); s.add_argument("--samples", type=int); s.set_defaults(fn=cmd_cycle)
    sub.add_parser("pin-evals").set_defaults(fn=cmd_pin)
    sub.add_parser("rollback").set_defaults(fn=cmd_rollback)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
