"""Thin wrapper around MLX-LM: ONE base model in memory, adapters toggled.

On a 16 GB M1 Pro a second copy of the 7B base for the judge is the
difference between fitting and swapping, so the base is loaded once and
shared. MLX applies a LoRA adapter by wrapping linear layers in-place; each
wrapper has a ``scale`` factor on the low-rank delta. Setting every scale to
0 is mathematically the frozen base, so:

  * ``student`` chat  -> adapter scales at their trained value
  * ``judge``   chat  -> adapter scales at 0 (frozen base, no adapter)

Keeping the judge frozen is what stops a candidate adapter from learning to
flatter its own grader.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .config import CFG, path

log = logging.getLogger(__name__)

try:  # MLX only exists on Apple Silicon; tests and dry runs work without it.
    from mlx_lm import generate as _mlx_generate
    from mlx_lm import load as _mlx_load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters as _load_adapters

    HAS_MLX = True
except ImportError:  # pragma: no cover
    HAS_MLX = False


class SharedBase:
    """The one loaded base model plus whichever adapter is currently applied."""

    _instance: "SharedBase | None" = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.adapter_dir: Optional[Path] = None
        self._lora_modules: list = []
        self._scales: list[float] = []

    @classmethod
    def get(cls) -> "SharedBase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        if not HAS_MLX:
            raise RuntimeError("mlx_lm is not installed; run on Apple Silicon or use --dry-run")
        log.info("loading base %s", CFG.model.base)
        self.model, self.tokenizer = _mlx_load(CFG.model.base)

    def apply_adapter(self, adapter_dir: Optional[Path]) -> None:
        """Apply an adapter. Switching adapters reloads the base (a few
        seconds); switching between adapter and no-adapter is instant."""
        self.ensure_loaded()
        if adapter_dir == self.adapter_dir:
            return
        if self.adapter_dir is not None:
            # Already wrapped with another adapter: start from a clean base.
            log.info("reloading base to switch adapter %s -> %s", self.adapter_dir, adapter_dir)
            self.model, self.tokenizer = _mlx_load(CFG.model.base)
            self._lora_modules, self._scales = [], []
        self.adapter_dir = None
        if adapter_dir and adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
            log.info("applying adapter %s", adapter_dir)
            self.model = _load_adapters(self.model, str(adapter_dir))
            self._lora_modules = [m for _, m in self.model.named_modules()
                                  if hasattr(m, "lora_a") and hasattr(m, "scale")]
            self._scales = [m.scale for m in self._lora_modules]
            self.adapter_dir = adapter_dir

    @contextmanager
    def adapter_disabled(self):
        """Temporarily zero every LoRA scale: the frozen base, for free."""
        for m in self._lora_modules:
            m.scale = 0.0
        try:
            yield
        finally:
            for m, s in zip(self._lora_modules, self._scales):
                m.scale = s

    def generate(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        return _mlx_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens,
                             sampler=make_sampler(temp=temperature), verbose=False)


class LocalModel:
    """A view onto the shared base: with an adapter (student) or without (judge)."""

    def __init__(self, adapter_dir: Optional[Path] = None, name: str = "model"):
        self.name = name
        self.adapter_dir = adapter_dir
        self.frozen = adapter_dir is None

    def load(self) -> "LocalModel":
        base = SharedBase.get()
        base.ensure_loaded()
        if not self.frozen:
            base.apply_adapter(self.adapter_dir)
        return self

    def swap_adapter(self, adapter_dir: Optional[Path]) -> None:
        self.adapter_dir = adapter_dir
        self.frozen = adapter_dir is None
        self.load()

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> str:
        base = SharedBase.get()
        self.load()
        max_tokens = max_tokens or CFG.model.max_tokens
        if self.frozen:
            with base.adapter_disabled():
                return base.generate(messages, temperature, max_tokens)
        return base.generate(messages, temperature, max_tokens)


class DryRunModel(LocalModel):
    """Deterministic stand-in so the whole pipeline runs on any machine."""

    def load(self) -> "LocalModel":
        return self

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> str:
        last = messages[-1]["content"]
        # Echo a trivially checkable answer so scorers have something to grade.
        if "2+2" in last or "2 + 2" in last:
            return "<think>trivial</think>\n4"
        if "Propose one experiment" in last:
            return ('<think>dry</think>\n{"hypothesis": "dry-run card", "file": "harness/decoding.yaml", '
                    '"expected": "none", "new_content": "temperature: 0.1\\nprefill: \\"\\"\\n"}')
        return "<think>dry run</think>\nDRY_RUN_RESPONSE"


def champion_model(dry_run: bool = False) -> LocalModel:
    cls = DryRunModel if dry_run else LocalModel
    return cls(adapter_dir=path(CFG.paths.adapters) / "champion", name="champion")


def candidate_model(adapter_dir: Path, dry_run: bool = False) -> LocalModel:
    cls = DryRunModel if dry_run else LocalModel
    return cls(adapter_dir=adapter_dir, name=adapter_dir.name)


def judge_model(dry_run: bool = False) -> LocalModel:
    cls = DryRunModel if dry_run else LocalModel
    return cls(adapter_dir=None, name="judge")


def strip_think(text: str) -> str:
    """DeepSeek-R1 distills emit <think>...</think>; scorers grade what follows."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()
