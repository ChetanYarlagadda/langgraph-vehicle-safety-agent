"""Provider-pluggable chat models with JSON-schema outputs and an on-disk response cache.

Default: local Ollama (``qwen3:4b``). OpenAI or Azure OpenAI can be selected through environment variables
(``SAFETYAGENT_LLM=openai|azure``); neither is needed to reproduce the results. If the ``langsmith`` package is
installed and ``LANGSMITH_TRACING=true`` (plus ``LANGSMITH_API_KEY``) is set, every call and every graph node is
traced; otherwise tracing is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import requests

try:  # optional tracing
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore[no-redef]
        return lambda f: f


@dataclass
class ChatReply:
    text: str
    latency_s: float  # model time of the original call (also reported for cache hits)
    cached: bool = False
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> ChatReply: ...


class OllamaChat:
    """Ollama ``/api/chat`` with structured output (``format`` = JSON schema), greedy decoding and thinking off.

    The local ``qwen3:4b`` tag is the Thinking-2507 build, whose template always opens a ``<think>`` block; with
    ``think: false`` plus a JSON-schema ``format`` the constrained decoder emits the JSON object directly.
    """

    def __init__(self, model: str = "qwen3:4b", host: str = "http://localhost:11434", num_ctx: int = 8192,
                 timeout: int = 300) -> None:
        self.model = model
        self.host = host if host.startswith("http") else f"http://{host}"
        self.name = f"ollama:{model}"
        self.options = {"temperature": 0.0, "seed": 0, "num_ctx": num_ctx}
        self.timeout = timeout

    @traceable(run_type="llm", name="ollama_chat")
    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> ChatReply:
        body = {"model": self.model, "messages": messages, "format": schema, "think": False, "stream": False,
                "keep_alive": "10m", "options": {**self.options, "num_predict": max_tokens}}
        t0 = time.perf_counter()
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        if "error" in out:
            raise RuntimeError(out["error"])
        return ChatReply(out["message"]["content"], time.perf_counter() - t0, False, out.get("prompt_eval_count"), out.get("eval_count"))


class OpenAIChat:
    """OpenAI or Azure OpenAI chat completions with a JSON-schema response format (optional provider)."""

    def __init__(self, model: str, azure: bool = False) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the 'openai' package to use the OpenAI / Azure OpenAI provider") from exc
        if azure:
            self.client = openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("OPENAI_API_VERSION", "2024-10-21"),
            )
        else:
            self.client = openai.OpenAI()
        self.model = model
        self.name = f"{'azure' if azure else 'openai'}:{model}"
        self.options = {"temperature": 0.0}

    @traceable(run_type="llm", name="openai_chat")
    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> ChatReply:  # pragma: no cover - needs a key
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0, max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema}},
        )
        u = resp.usage
        return ChatReply(resp.choices[0].message.content or "", time.perf_counter() - t0, False,
                         u.prompt_tokens if u else None, u.completion_tokens if u else None)


class CachedChat:
    """Wraps a ChatModel; replies are stored in SQLite under sha256(model, options, messages, schema, max_tokens)."""

    def __init__(self, inner: ChatModel, path: Path) -> None:
        self.inner = inner
        self.name = inner.name
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, model TEXT, request TEXT, text TEXT, latency_s REAL, "
            "prompt_tokens INTEGER, output_tokens INTEGER, created_utc TEXT)"
        )
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.new_seconds = 0.0

    def key(self, messages: list[dict], schema: dict, max_tokens: int) -> tuple[str, str]:
        request = json.dumps({"model": self.inner.name, "options": getattr(self.inner, "options", {}), "messages": messages,
                              "schema": schema, "max_tokens": max_tokens}, sort_keys=True)
        return hashlib.sha256(request.encode()).hexdigest(), request

    def chat(self, messages: list[dict], schema: dict, max_tokens: int) -> ChatReply:
        key, request = self.key(messages, schema, max_tokens)
        with self.lock:
            row = self.db.execute("SELECT text, latency_s, prompt_tokens, output_tokens FROM replies WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return ChatReply(row[0], row[1], True, row[2], row[3])
        reply = self.inner.chat(messages, schema, max_tokens)
        self.misses += 1
        self.new_seconds += reply.latency_s
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (key, self.inner.name, request, reply.text, reply.latency_s, reply.prompt_tokens,
                             reply.output_tokens, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            self.db.commit()
        return reply

    def total_model_seconds(self) -> float:
        """Model time of every reply ever stored in this cache (all runs, all experiments)."""
        with self.lock:
            return float(self.db.execute("SELECT coalesce(sum(latency_s), 0) FROM replies").fetchone()[0])


def make_llm_from_env(cache_path: Path | None = None) -> ChatModel:
    """Chat model chosen by ``SAFETYAGENT_LLM`` (ollama | openai | azure), optionally wrapped in the disk cache."""
    provider = os.getenv("SAFETYAGENT_LLM", "ollama").lower()
    if provider == "ollama":
        llm: ChatModel = OllamaChat(os.getenv("SAFETYAGENT_OLLAMA_MODEL", "qwen3:4b"), os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    elif provider == "openai":
        llm = OpenAIChat(os.getenv("SAFETYAGENT_OPENAI_MODEL", "gpt-4o-mini"))
    elif provider == "azure":
        llm = OpenAIChat(os.environ["AZURE_OPENAI_DEPLOYMENT"], azure=True)
    else:
        raise ValueError(f"unknown SAFETYAGENT_LLM provider '{provider}' (use ollama, openai or azure)")
    return CachedChat(llm, cache_path) if cache_path is not None else llm
