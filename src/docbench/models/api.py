"""OpenAI-compatible HTTP API provider.

Works with OpenAI, Azure OpenAI, vLLM, Ollama, Together, Fireworks, Groq,
Anthropic (via compatible proxies), and any `/v1/chat/completions` endpoint.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from docbench.models.base import ModelProvider, ModelRequest, ModelResponse


def _encode_image(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, data


class APIModelProvider(ModelProvider):
    """Call a remote chat-completions API."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 120.0,
        headers: dict[str, str] | None = None,
        request_path: str = "/chat/completions",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        import os

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env)
        self.timeout_s = timeout_s
        self.request_path = request_path
        self._extra_headers = headers or {}
        self._client = httpx.Client(timeout=timeout_s)
        self._async_client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self._extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _build_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})

        if request.images:
            content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
            for img in request.images:
                mime, data = _encode_image(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": request.prompt})
        return messages

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        payload.update(request.extra)
        payload.update(self.kwargs.get("extra_body", {}))
        return payload

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> ModelResponse:
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            return ModelResponse(
                text="",
                latency_ms=latency_ms,
                model_id=self.model_id,
                raw=data,
                error=f"Unexpected response shape: {exc}",
            )
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text.strip(),
            latency_ms=latency_ms,
            model_id=self.model_id,
            usage=usage,
            raw=data,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, request: ModelRequest) -> ModelResponse:
        url = f"{self.base_url}{self.request_path}"
        payload = self._build_payload(request)
        start = time.perf_counter()
        try:
            resp = self._client.post(url, json=payload, headers=self._headers())
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            return self._parse_response(resp.json(), latency_ms)
        except Exception as exc:  # noqa: BLE001 — surface to benchmark report
            latency_ms = (time.perf_counter() - start) * 1000
            return ModelResponse(
                text="",
                latency_ms=latency_ms,
                model_id=self.model_id,
                error=str(exc),
            )

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout_s)
        url = f"{self.base_url}{self.request_path}"
        payload = self._build_payload(request)
        start = time.perf_counter()
        try:
            resp = await self._async_client.post(url, json=payload, headers=self._headers())
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            return self._parse_response(resp.json(), latency_ms)
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            return ModelResponse(
                text="",
                latency_ms=latency_ms,
                model_id=self.model_id,
                error=str(exc),
            )

    def close(self) -> None:
        self._client.close()
        if self._async_client is not None:
            # Sync close of async client is best-effort in sync contexts
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.aclose())
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except Exception:  # noqa: BLE001
                pass
            self._async_client = None