"""YAML/JSON config loading and validation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return os.environ.get(key, match.group(0))

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class ModelConfig(BaseModel):
    type: Literal["api", "local", "mock"] = "api"
    id: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 120.0
    headers: dict[str, str] | None = None
    request_path: str = "/chat/completions"
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    max_new_tokens: int = 1024
    options: dict[str, Any] = Field(default_factory=dict)


class QualitativeTaskConfig(BaseModel):
    enabled: bool = True
    dataset: str
    max_samples: int | None = None
    metrics: list[str] = Field(default_factory=lambda: ["exact_match", "token_f1"])
    system_prompt: str | None = None
    user_prompt_template: str | None = None


class QualitativeConfig(BaseModel):
    enabled: bool = True
    tasks: dict[str, QualitativeTaskConfig] = Field(default_factory=dict)


class StressWorkloadConfig(BaseModel):
    enabled: bool = True
    modality: Literal["text", "image"] = "text"
    dataset: str
    # Latency
    warmup: int = 2
    iterations: int = 20
    # Concurrency
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    requests_per_level: int = 32
    prompt: str | None = None


class StressConfig(BaseModel):
    enabled: bool = True
    workloads: dict[str, StressWorkloadConfig] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    dir: str = "results"
    formats: list[str] = Field(default_factory=lambda: ["json", "markdown"])
    run_name: str | None = None


class BenchConfig(BaseModel):
    """Top-level benchmark configuration."""

    model: ModelConfig
    qualitative: QualitativeConfig = Field(default_factory=QualitativeConfig)
    stress: StressConfig = Field(default_factory=StressConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = 42

    @field_validator("model", mode="before")
    @classmethod
    def _coerce_model(cls, v: Any) -> Any:
        return v


def load_config(path: str | Path) -> BenchConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        # Try YAML first
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    data = _expand_env(data)
    return BenchConfig.model_validate(data)


def config_to_dict(cfg: BenchConfig) -> dict[str, Any]:
    return cfg.model_dump()