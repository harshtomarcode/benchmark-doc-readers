"""Local model provider via Hugging Face Transformers.

Requires optional extras: `pip install -e ".[local]"`.
Supports text-only and vision-language models that expose a chat/processor API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from docbench.models.base import ModelProvider, ModelRequest, ModelResponse


class LocalModelProvider(ModelProvider):
    """Load a local / Hugging Face model for offline evaluation."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._is_vlm = False

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local models require optional deps. Install with: pip install -e '.[local]'"
            ) from exc

        dtype = getattr(torch, self.torch_dtype, "auto") if self.torch_dtype != "auto" else "auto"

        # Prefer processor (VLM); fall back to tokenizer (text-only).
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            self._is_vlm = True
        except Exception:  # noqa: BLE001
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            self._is_vlm = False

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def supports_images(self) -> bool:
        self._lazy_load()
        return self._is_vlm

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._lazy_load()
        import torch
        from PIL import Image

        start = time.perf_counter()
        try:
            max_tokens = request.max_tokens or self.max_new_tokens
            if self._is_vlm and request.images:
                images = [Image.open(Path(p)).convert("RGB") for p in request.images]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"} for _ in images
                        ]
                        + [{"type": "text", "text": request.prompt}],
                    }
                ]
                # Many VLMs expect apply_chat_template; fall back to raw prompt.
                processor = self._processor
                if hasattr(processor, "apply_chat_template"):
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = request.prompt
                inputs = processor(text=[text], images=images, return_tensors="pt")
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    out = self._model.generate(**inputs, max_new_tokens=max_tokens)
                # Decode only newly generated tokens when possible
                gen = out[:, inputs["input_ids"].shape[-1] :]
                text_out = processor.batch_decode(gen, skip_special_tokens=True)[0]
            else:
                tokenizer = self._tokenizer or self._processor
                prompt = request.prompt
                if request.system:
                    prompt = f"{request.system}\n\n{prompt}"
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    out = self._model.generate(**inputs, max_new_tokens=max_tokens)
                gen = out[:, inputs["input_ids"].shape[-1] :]
                text_out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

            latency_ms = (time.perf_counter() - start) * 1000
            return ModelResponse(
                text=text_out.strip(),
                latency_ms=latency_ms,
                model_id=self.model_id,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            return ModelResponse(
                text="",
                latency_ms=latency_ms,
                model_id=self.model_id,
                error=str(exc),
            )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass