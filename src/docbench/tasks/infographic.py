"""Infographic understanding (image → text)."""

from __future__ import annotations

from docbench.models.base import ModelResponse
from docbench.tasks.base import QualitativeTask, Sample


class InfographicTask(QualitativeTask):
    name = "infographic"
    requires_images = True

    def postprocess(self, response: ModelResponse, sample: Sample) -> str:
        return response.text.strip()