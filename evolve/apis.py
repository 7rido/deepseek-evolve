"""Frontier-model clients: DeepSeek (OpenAI-compatible) and Anthropic.

Stdlib only, so the repo has no SDK version drift. Every call returns the
response text and a ``source`` tag the rest of the pipeline uses to decide
what the text may be used for:

  source == "deepseek-api"   -> may become training data (teacher, critic)
  source == "anthropic-api"  -> research only; NEVER training data

That rule is enforced in trajectories.build_sft_dataset, not just here.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import env
from .config import CFG

log = logging.getLogger(__name__)

DEEPSEEK = "deepseek-api"
ANTHROPIC = "anthropic-api"
LOCAL = "local"


@dataclass
class Reply:
    text: str
    source: str
    model: str
    usage: dict


class APIError(RuntimeError):
    pass


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=CFG.apis.timeout_seconds) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise APIError(f"{url} -> {e.code}: {e.read()[:500].decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise APIError(f"{url} -> {e.reason}") from e


# -- DeepSeek -----------------------------------------------------------

def deepseek_available() -> bool:
    return env.deepseek_key() is not None


def deepseek_chat(messages: list[dict], model: str | None = None,
                  temperature: float = 0.6, max_tokens: int = 8192) -> Reply:
    key = env.deepseek_key()
    if not key:
        raise APIError("DEEPSEEK_API_KEY not set (put it in .env)")
    model = model or CFG.apis.deepseek_teacher_model
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if "reasoner" not in model:  # R1 ignores temperature; V3 honors it
        body["temperature"] = temperature
    data = _post(f"{CFG.apis.deepseek_base_url}/chat/completions",
                 {"Authorization": f"Bearer {key}"}, body)
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    # deepseek-reasoner returns its chain of thought separately; keep it in
    # <think> tags so the student learns the same shape it already emits.
    if msg.get("reasoning_content"):
        text = f"<think>\n{msg['reasoning_content']}\n</think>\n{text}"
    return Reply(text, DEEPSEEK, model, data.get("usage", {}))


# -- Anthropic ----------------------------------------------------------

def anthropic_available() -> bool:
    return env.anthropic_key() is not None


def claude_chat(messages: list[dict], system: str = "", model: str | None = None,
                max_tokens: int | None = None) -> Reply:
    key = env.anthropic_key()
    if not key:
        raise APIError("ANTHROPIC_API_KEY not set (put it in .env)")
    model = model or CFG.apis.anthropic_model
    body = {"model": model, "max_tokens": max_tokens or CFG.apis.anthropic_max_tokens,
            "messages": messages}
    if system:
        body["system"] = system
    data = _post("https://api.anthropic.com/v1/messages",
                 {"x-api-key": key, "anthropic-version": "2023-06-01"}, body)
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return Reply(text, ANTHROPIC, model, data.get("usage", {}))
