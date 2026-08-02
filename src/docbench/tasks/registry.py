"""Task name → class registry."""

from __future__ import annotations

from typing import Type

from docbench.tasks.base import QualitativeTask
from docbench.tasks.chart_reading import ChartReadingTask
from docbench.tasks.infographic import InfographicTask
from docbench.tasks.ocr import OCRTask
from docbench.tasks.table_reading import TableReadingTask
from docbench.tasks.text_reading import TextReadingTask

TASK_REGISTRY: dict[str, Type[QualitativeTask]] = {
    "text_reading": TextReadingTask,
    "table_reading": TableReadingTask,
    "chart_reading": ChartReadingTask,
    "infographic": InfographicTask,
    "ocr": OCRTask,
}


def get_task_class(name: str) -> Type[QualitativeTask]:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {name!r}. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]