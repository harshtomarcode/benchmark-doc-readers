"""Deterministic mock provider for dry-runs and unit tests."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from docbench.models.base import ModelProvider, ModelRequest, ModelResponse


class MockModelProvider(ModelProvider):
    """Returns canned answers with configurable latency."""

    def __init__(
        self,
        model_id: str = "mock",
        *,
        latency_ms: float = 50.0,
        answers: dict[str, str] | None = None,
        default_answer: str = "Mock answer",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.latency_ms = latency_ms
        self.answers = answers or {}
        self.default_answer = default_answer

    def _lookup(self, request: ModelRequest) -> str:
        # Match by exact prompt, then by any image stem, then default.
        if request.prompt in self.answers:
            return self.answers[request.prompt]
        for img in request.images:
            key = str(img)
            if key in self.answers:
                return self.answers[key]
            stem = str(img).rsplit("/", 1)[-1]
            if stem in self.answers:
                return self.answers[stem]
        return self.default_answer

    def generate(self, request: ModelRequest) -> ModelResponse:
        time.sleep(self.latency_ms / 1000.0)
        return ModelResponse(
            text=self._lookup(request),
            latency_ms=self.latency_ms,
            model_id=self.model_id,
            usage={"prompt_tokens": 10, "completion_tokens": 10},
        )

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(self.latency_ms / 1000.0)
        return ModelResponse(
            text=self._lookup(request),
            latency_ms=self.latency_ms,
            model_id=self.model_id,
            usage={"prompt_tokens": 10, "completion_tokens": 10},
        )