# Datasets

Each subdirectory is a task dataset consumed by the qualitative (and optionally stress) runners.

| Directory | Task | Contents |
|-----------|------|----------|
| `text/` | `text_reading` | Plain-text documents + QA |
| `tables/` | `table_reading` | CSV / Markdown tables + one table image |
| `charts/` | `chart_reading` | Bar chart images + QA |
| `infographics/` | `infographic` | Infographic images + QA |
| `ocr/` | `ocr` | Noisy “scanned” document images + QA |

## Adding samples

Append a line to `samples.jsonl`:

```json
{"id": "unique-id", "document": "file.txt", "question": "...", "answer": "..."}
```

For image tasks use `"image": "file.png"` (or `document` / `path`). Paths are relative to the dataset directory.

Regenerate the bundled fixtures:

```bash
python scripts/generate_sample_datasets.py
```