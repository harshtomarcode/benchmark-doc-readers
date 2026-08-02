"""Build a ModelProvider from config dict."""

from __future__ import annotations

from typing import Any

from docbench.models.api import APIModelProvider
from docbench.models.base import ModelProvider
from docbench.models.local import LocalModelProvider


def build_provider(model_cfg: dict[str, Any]) -> ModelProvider:
    """Instantiate a provider from the `model:` section of a config file.

    Supported types:
      - api   : OpenAI-compatible HTTP API
      - local : Hugging Face / local Transformers
      - mock  : Deterministic stub for dry-runs and unit tests
    """
    kind = (model_cfg.get("type") or "api").lower()
    model_id = model_cfg.get("id") or model_cfg.get("model_id") or "unknown"
    options = dict(model_cfg.get("options") or {})

    if kind == "api":
        return APIModelProvider(
            model_id=model_id,
            base_url=model_cfg.get("base_url", "https://api.openai.com/v1"),
            api_key=model_cfg.get("api_key"),
            api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
            timeout_s=float(model_cfg.get("timeout_s", 120)),
            headers=model_cfg.get("headers"),
            request_path=model_cfg.get("request_path", "/chat/completions"),
            **options,
        )
    if kind == "local":
        return LocalModelProvider(
            model_id=model_id,
            device=model_cfg.get("device", "auto"),
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 1024)),
            **options,
        )
    if kind == "mock":
        from docbench.models.mock import MockModelProvider

        return MockModelProvider(model_id=model_id, **options)

    raise ValueError(f"Unknown model type: {kind!r}. Use 'api', 'local', or 'mock'.")