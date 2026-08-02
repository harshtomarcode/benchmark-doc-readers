"""OCR / scanned document reading."""

from __future__ import annotations

from docbench.models.base import ModelResponse
from docbench.tasks.base import QualitativeTask, Sample


class OCRTask(QualitativeTask):
    name = "ocr"
    requires_images = True

    def postprocess(self, response: ModelResponse, sample: Sample) -> str:
        return response.text.strip()