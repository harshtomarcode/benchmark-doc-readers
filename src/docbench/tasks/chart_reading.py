"""Chart reading (image → text / numeric QA)."""

from __future__ import annotations

from docbench.models.base import ModelResponse
from docbench.tasks.base import QualitativeTask, Sample


class ChartReadingTask(QualitativeTask):
    name = "chart_reading"
    requires_images = True

    def postprocess(self, response: ModelResponse, sample: Sample) -> str:
        return response.text.strip()