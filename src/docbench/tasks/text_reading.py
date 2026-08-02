"""Pure text document reading / QA."""

from __future__ import annotations

from docbench.models.base import ModelResponse
from docbench.tasks.base import QualitativeTask, Sample


class TextReadingTask(QualitativeTask):
    name = "text_reading"
    requires_images = False

    def postprocess(self, response: ModelResponse, sample: Sample) -> str:
        return response.text.strip()