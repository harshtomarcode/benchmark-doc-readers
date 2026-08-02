"""Base qualitative task runner."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from docbench.evaluators.metrics import aggregate, score
from docbench.models.base import ModelProvider, ModelRequest, ModelResponse

console = Console()


@dataclass
class Sample:
    id: str
    question: str
    answer: str
    document_path: str | Path | None = None
    document_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    sample_id: str
    prediction: str
    reference: str
    scores: dict[str, float]
    latency_ms: float
    error: str | None = None
    raw_response: str | None = None


@dataclass
class TaskResult:
    task_name: str
    samples: list[SampleResult]
    metrics: dict[str, float]
    n_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "metrics": self.metrics,
            "n_samples": len(self.samples),
            "n_errors": self.n_errors,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "prediction": s.prediction,
                    "reference": s.reference,
                    "scores": s.scores,
                    "latency_ms": s.latency_ms,
                    "error": s.error,
                }
                for s in self.samples
            ],
        }


DEFAULT_PROMPTS: dict[str, str] = {
    "text_reading": (
        "Read the following document carefully and answer the question.\n\n"
        "Document:\n{document}\n\nQuestion: {question}\n\n"
        "Answer concisely with only the final answer."
    ),
    "table_reading": (
        "You are given a table (as text or image). Answer the question based only on the table.\n\n"
        "{document}\n\nQuestion: {question}\n\n"
        "Answer concisely with only the final answer."
    ),
    "chart_reading": (
        "Examine the chart image and answer the question.\n\n"
        "Question: {question}\n\n"
        "Answer concisely with only the final answer (numbers when applicable)."
    ),
    "infographic": (
        "Examine the infographic and answer the question based on its content.\n\n"
        "Question: {question}\n\n"
        "Answer concisely with only the final answer."
    ),
    "ocr": (
        "Transcribe or answer based on the scanned document image.\n\n"
        "Question: {question}\n\n"
        "Answer with the requested text exactly as it appears when possible."
    ),
}


class QualitativeTask(ABC):
    """A single qualitative document-reading task."""

    name: str = "base"
    requires_images: bool = False

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        metrics: list[str] | None = None,
        max_samples: int | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.metrics = metrics or ["exact_match", "token_f1"]
        self.max_samples = max_samples
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template or DEFAULT_PROMPTS.get(
            self.name, "Document:\n{document}\n\nQuestion: {question}"
        )

    def load_samples(self) -> list[Sample]:
        manifest = self.dataset_path / "samples.jsonl"
        if not manifest.exists():
            # Allow a single JSON list file
            alt = self.dataset_path / "samples.json"
            if not alt.exists():
                raise FileNotFoundError(
                    f"No samples.jsonl or samples.json in {self.dataset_path}"
                )
            data = json.loads(alt.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("samples", [])
        else:
            rows = []
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        samples: list[Sample] = []
        for row in rows:
            doc_path = row.get("document") or row.get("image") or row.get("path")
            if doc_path and not Path(doc_path).is_absolute():
                doc_path = str(self.dataset_path / doc_path)
            samples.append(
                Sample(
                    id=str(row.get("id", len(samples))),
                    question=row["question"],
                    answer=str(row["answer"]),
                    document_path=doc_path,
                    document_text=row.get("text") or row.get("document_text"),
                    meta={k: v for k, v in row.items() if k not in {
                        "id", "question", "answer", "document", "image", "path", "text",
                        "document_text",
                    }},
                )
            )
        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def build_prompt(self, sample: Sample) -> str:
        document = sample.document_text or ""
        if not document and sample.document_path and not self.requires_images:
            p = Path(sample.document_path)
            if p.suffix.lower() in {".txt", ".md", ".csv", ".tsv", ".html"}:
                document = p.read_text(encoding="utf-8")
        return self.user_prompt_template.format(
            document=document,
            question=sample.question,
        )

    def build_request(self, sample: Sample) -> ModelRequest:
        images: list[str] = []
        if self.requires_images and sample.document_path:
            images = [str(sample.document_path)]
        elif sample.document_path and Path(sample.document_path).suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp",
        }:
            images = [str(sample.document_path)]
        return ModelRequest(
            prompt=self.build_prompt(sample),
            images=images,
            system=self.system_prompt,
        )

    @abstractmethod
    def postprocess(self, response: ModelResponse, sample: Sample) -> str:
        """Extract the final answer string from the model response."""

    def run(self, provider: ModelProvider) -> TaskResult:
        samples = self.load_samples()
        results: list[SampleResult] = []
        n_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(f"[cyan]{self.name}", total=len(samples))
            for sample in samples:
                req = self.build_request(sample)
                resp = provider.generate(req)
                if not resp.ok:
                    n_errors += 1
                    results.append(
                        SampleResult(
                            sample_id=sample.id,
                            prediction="",
                            reference=sample.answer,
                            scores={m: 0.0 for m in self.metrics},
                            latency_ms=resp.latency_ms,
                            error=resp.error,
                        )
                    )
                else:
                    pred = self.postprocess(resp, sample)
                    results.append(
                        SampleResult(
                            sample_id=sample.id,
                            prediction=pred,
                            reference=sample.answer,
                            scores=score(pred, sample.answer, self.metrics),
                            latency_ms=resp.latency_ms,
                            raw_response=resp.text,
                        )
                    )
                progress.advance(task_id)

        metrics = aggregate([r.scores for r in results])
        avg_latency = (
            sum(r.latency_ms for r in results) / len(results) if results else 0.0
        )
        metrics["avg_latency_ms"] = avg_latency
        return TaskResult(
            task_name=self.name,
            samples=results,
            metrics=metrics,
            n_errors=n_errors,
        )