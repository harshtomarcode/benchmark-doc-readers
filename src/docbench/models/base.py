"""Abstract model provider interface.

Any backend (OpenAI-compatible API, local Transformers, custom) implements
`ModelProvider` so qualitative and stress tasks stay backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass
class MessagePart:
    """A single content part: text and/or an image path/URL."""

    type: str  # "text" | "image"
    text: str | None = None
    image_path: str | Path | None = None
    image_url: str | None = None


@dataclass
class ModelRequest:
    """Normalized request sent to any provider."""

    prompt: str
    images: Sequence[str | Path] = field(default_factory=tuple)
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def modality(self) -> Modality:
        if self.images and self.prompt:
            return Modality.MIXED
        if self.images:
            return Modality.IMAGE
        return Modality.TEXT


@dataclass
class ModelResponse:
    """Normalized response from any provider."""

    text: str
    latency_ms: float
    model_id: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelProvider(ABC):
    """Pluggable model interface used by all benchmark tasks."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Synchronous generation (used by qualitative + latency tests)."""

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        """Async generation (used by concurrency stress tests).

        Default falls back to sync `generate` in a thread. Override for
        native async HTTP clients.
        """
        import asyncio

        return await asyncio.to_thread(self.generate, request)

    def supports_images(self) -> bool:
        """Whether this provider can accept image inputs."""
        return True

    def close(self) -> None:
        """Release resources (HTTP clients, GPU memory, etc.)."""

    def __enter__(self) -> ModelProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()