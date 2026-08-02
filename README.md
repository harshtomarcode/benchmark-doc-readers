# benchmark-doc-readers

Config-driven benchmark suite for **document reading** models — qualitative accuracy and stress (latency / concurrency).

Point a YAML config at a model (local weights or an OpenAI-compatible API). All tasks plug in from that config.

## What it measures

### Qualitative

| Task | Modality | What it tests |
|------|----------|---------------|
| `text_reading` | text | Pure document QA |
| `table_reading` | text / image | Tables (CSV/Markdown or rendered images) |
| `chart_reading` | image → text | Chart understanding / numeric QA |
| `infographic` | image → text | Dense visual + text layouts |
| `ocr` | image → text | Scanned / noisy documents |

Metrics (selectable per task): `exact_match`, `contains`, `token_f1`, `json_exact`, `numeric_tolerance`.

### Stress

For **pure text** and **image-to-text** workloads:

- **Latency** — warmup + N timed requests → mean / p50 / p90 / p95 / p99
- **Concurrency** — sweep concurrency levels → throughput (rps) and latency under load

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Smoke test with the built-in mock provider (no API key)
docbench run -c configs/example_mock.yaml
```

Results land in `results/<run_name>/` as `results.json` and `report.md`.

### Against an API

```bash
export OPENAI_API_KEY=sk-...
docbench run -c configs/example_api.yaml
```

Works with any OpenAI-compatible chat endpoint (OpenAI, Azure, vLLM, Ollama, Groq, etc.) — set `model.base_url` and `model.id`.

### Against a local model

```bash
pip install -e ".[local]"
docbench run -c configs/example_local.yaml
```

## Config shape

```yaml
model:
  type: api          # api | local | mock
  id: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY

qualitative:
  enabled: true
  tasks:
    text_reading:
      enabled: true
      dataset: datasets/text
      metrics: [exact_match, contains, token_f1]
    chart_reading:
      enabled: true
      dataset: datasets/charts
      metrics: [contains, token_f1]

stress:
  enabled: true
  workloads:
    pure_text:
      modality: text
      dataset: datasets/text
      warmup: 2
      iterations: 20
      concurrency_levels: [1, 4, 8, 16]
      requests_per_level: 32
    image_to_text:
      modality: image
      dataset: datasets/charts
      warmup: 2
      iterations: 20
      concurrency_levels: [1, 4, 8]
      requests_per_level: 24

output:
  dir: results
  formats: [json, markdown]
```

`${ENV_VAR}` placeholders are expanded in the config. See `configs/` for full examples (`example_api.yaml`, `example_local.yaml`, `example_ollama.yaml`, `example_mock.yaml`).

## CLI

```bash
docbench list-tasks
docbench run -c configs/example_api.yaml
docbench run -c configs/example_api.yaml --qualitative-only
docbench run -c configs/example_api.yaml --stress-only --run-name my_run
```

## Dataset format

Each task dataset is a directory with `samples.jsonl` (or `samples.json`):

```json
{"id": "text-001", "document": "q1_report.txt", "question": "What was revenue?", "answer": "$12.4 million"}
{"id": "chart-001", "image": "revenue_by_region.png", "question": "Highest region?", "answer": "East"}
```

- `document` / `image` / `path` — relative to the dataset directory
- Text tasks also accept inline `text` / `document_text`
- Bundled samples live under `datasets/{text,tables,charts,infographics,ocr}/`
- Regenerate them with: `python scripts/generate_sample_datasets.py`

## Architecture

```
config YAML
    │
    ▼
ModelProvider  ◄── api | local | mock   (plug in your own)
    │
    ├── QualitativeTask[]  (text, table, chart, infographic, ocr)
    │         └── metrics
    └── Stress
              ├── latency
              └── concurrency
                        │
                        ▼
                   results/ + report.md
```

Implement `docbench.models.base.ModelProvider` and register it in `models/factory.py` to add a new backend. Add a class under `tasks/` and register it in `tasks/registry.py` for a new qualitative task.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT