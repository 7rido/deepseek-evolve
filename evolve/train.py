"""Loop A, steps 4-6: train a LoRA candidate with mlx_lm, gate it, promote or reject.

mlx_lm.lora is invoked as a subprocess rather than imported so a training
crash (OOM is the usual one on 16 GB) cannot take the orchestrator down.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from . import adapters, evals, notebook
from .config import CFG, path
from .model import candidate_model, champion_model, judge_model
from .trajectories import build_sft_dataset

log = logging.getLogger(__name__)


def train_candidate(from_base: bool = False, dry_run: bool = False) -> Path | None:
    n_train, n_valid = build_sft_dataset()
    if n_train < CFG.filter.min_examples_to_train and not dry_run:
        log.warning("skipping training: %d examples < %d", n_train,
                    CFG.filter.min_examples_to_train)
        return None

    out = path(CFG.paths.adapters) / f"candidate-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm.lora",
           "--model", CFG.model.base,
           "--train",
           "--data", str(path(CFG.paths.sft)),
           "--adapter-path", str(out),
           "--num-layers", str(CFG.train.lora_layers),
           "--iters", str(CFG.train.iters),
           "--learning-rate", str(CFG.train.learning_rate),
           "--batch-size", str(CFG.train.batch_size),
           "--max-seq-length", str(CFG.train.max_seq_length),
           "--grad-checkpoint"]
    # Stack on the champion unless we are resetting from base; stacking keeps
    # each cycle short, the reset prevents drift compounding forever.
    champ = adapters.champion_dir()
    if champ.exists() and not from_base:
        cmd += ["--resume-adapter-file", str(champ / "adapters.safetensors")]

    if dry_run:
        (out / "adapters.safetensors").write_bytes(b"")
        (out / "adapter_config.json").write_text("{}")
        log.info("dry run: would run %s", " ".join(cmd))
        return out

    log.info("training: %s", " ".join(cmd))
    p = subprocess.run(cmd, cwd=path("."), text=True)
    if p.returncode != 0:
        log.error("mlx_lm.lora failed with code %d", p.returncode)
        return None
    return out


def gate_and_promote(candidate: Path, dry_run: bool = False) -> bool:
    judge = judge_model(dry_run)
    cand_model = candidate_model(candidate, dry_run)
    ok, why, rep = evals.gate(cand_model, judge, candidate.name, adapters.champion_score())
    if ok:
        adapters.promote(candidate, rep.mean)
        log.info("PROMOTED %s (%.1f): %s", candidate.name, rep.mean, why)
    else:
        adapters.reject(candidate, rep.mean, why)
        log.info("rejected %s (%.1f): %s", candidate.name, rep.mean, why)
    notebook.write_entry("Loop A", [{"id": candidate.name, "status": "promoted" if ok else "rejected",
                                     "score": rep.mean, "reason": why,
                                     "must_pass_failures": rep.must_pass_failures}])
    return ok


def cycle(dry_run: bool = False) -> bool:
    """One full Loop A cycle. Collection is a separate command so you can
    run it overnight and train in the morning."""
    reset = adapters.generations_from_base() >= CFG.train.max_generations_from_base
    if reset:
        log.info("generation cap reached: retraining from frozen base on full dataset")
    cand = train_candidate(from_base=reset, dry_run=dry_run)
    if cand is None:
        return False
    promoted = gate_and_promote(cand, dry_run=dry_run)
    if promoted and reset:
        adapters.reset_generations()
    return promoted
