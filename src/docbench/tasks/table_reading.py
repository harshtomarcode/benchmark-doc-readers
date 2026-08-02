"""Table reading — supports text tables and table images."""

from __future__ import annotations

from docbench.models.base import ModelResponse
from docbench.tasks.base import QualitativeTask, Sample


class TableReadingTask(QualitativeTask):
    name = "table_reading"
    requires_images = False  # may still attach images when present

    def postprocess(self, response: ModelResponse, sample: Sample) -> str:
        return response.text.strip()