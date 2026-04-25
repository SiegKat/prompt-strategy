"""Thin wrapper around the Ollama Python client.

Exposes a single entrypoint (`OllamaRunner.run`) that mirrors the
`model_request` helper from the original notebook but returns a
structured `ModelResponse` instead of a tuple.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


DEFAULT_MODEL = "tinyllama"


@dataclass
class ModelResponse:
    text: str
    tokens_used: int
    latency_ms: float
    model: str


class OllamaRunner:
    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None) -> None:
        from ollama import Client  # lazy: keeps strategy tests runnable without the dep

        self.model = model
        self.client = Client(host=host) if host else Client()

    def run(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        num_predict: int = 600,
        context_window: int = 2048,
        model: str | None = None,
    ) -> ModelResponse:
        target_model = model or self.model
        messages = [{"role": "user", "content": prompt}]

        start = time.perf_counter()
        response = self.client.chat(
            model=target_model,
            messages=messages,
            options={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_predict": num_predict,
                "num_ctx": context_window,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        text = response["message"]["content"]
        tokens_used = response.get("eval_count", len(text.split()))

        return ModelResponse(
            text=text,
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
            model=target_model,
        )
