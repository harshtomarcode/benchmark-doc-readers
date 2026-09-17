# benchmark-doc-readers

Config-driven benchmark suite for **document reading** models — qualitative accuracy and stress (latency / concurrency).

Compare document parsers on actual files, or run the existing model QA/stress suite.

## Compare document readers

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp env.example .env

# Check which tools will run. No imports of model packages, uploads, or inference.
docbench run-readers -c configs/readers.yaml --list

# After filling only the variables for tools you want to test:
docbench run-readers -c configs/readers.yaml
docbench run-readers -c configs/readers.yaml --tool docling --limit 1
```

Every assignment in `env.example` is empty. **An empty or whitespace-only gate
disables that tool**, even if it is installed. An all-empty run exits successfully
without opening the dataset or calling anything. For `run-readers`, explicit
entries in `.env` override exported environment variables, including blank values.
Use `--env-file path/to/file` for another file; omitted entries use the process environment.
The `--tool` filter never overrides an empty gate. Existing `docbench run` QA configs
keep their original behavior and do not use these reader gates.

| Reader name | Enable by setting | Implementation / optional installation |
|---|---|---|
| `llamaparse` | `LLAMA_CLOUD_API_KEY` | LlamaParse v2 upload, poll, Markdown and native result |
| `datalab` | `DATALAB_API_KEY` | Hosted Datalab Convert / Extract; distinct from local Marker |
| `reducto` | `REDUCTO_API_KEY` | Upload, asynchronous Parse / Extract, retrieve results |
| `mistral` | `MISTRAL_API_KEY` | File upload, OCR, optional document annotations |
| `unstructured_api` | `UNSTRUCTURED_API_KEY` | **Legacy Partition API**; optional `UNSTRUCTURED_API_URL` endpoint. This is not the managed Workflows VLM product. |
| `docling` | `DOCLING_PYTHON` | Interpreter with `docling` installed |
| `marker` | `MARKER_PYTHON` | Interpreter with `marker-pdf` and its documented Surya inference backend |
| `unstructured` | `UNSTRUCTURED_PYTHON` | Interpreter with `unstructured[all-docs]`; install the format/OCR system dependencies for your corpus |
| `mineru` | `MINERU_COMMAND` | Installed `mineru` CLI executable with the dependencies for the selected backend |
| `paddleocr` | `PADDLEOCR_PYTHON` | Interpreter with `paddleocr[doc-parser]` and the appropriate PaddlePaddle CPU/GPU runtime |
| `olmocr` | `OLMOCR_PYTHON` | Interpreter with `olmocr` and its inference/runtime dependencies, or a configured model server |
| `pymupdf4llm` | `PYMUPDF4LLM_PYTHON` | Interpreter with `pymupdf4llm` installed |
| `nuextract` | `NUEXTRACT_BASE_URL` | Existing NuExtract3-compatible OpenAI API endpoint ending in `/v1` |

Local Python variables take an interpreter path, **not** a boolean. For example,
install a chosen library in its own environment and set `DOCLING_PYTHON` to that
environment's absolute `bin/python` path. `MINERU_COMMAND` takes an executable path
without arguments. Libraries are intentionally isolated because their ML dependencies
can conflict; the main installation does not install or launch unused models.

For NuExtract, set its served model name in YAML. Optional `NUEXTRACT_PYTHON` chooses
the interpreter with `pymupdf` (PDF rendering) and `numind` (JSON Schema conversion);
otherwise the benchmark interpreter is used (use `numind>=0.3.1`). `NUEXTRACT_API_KEY` is optional for
unauthenticated local servers. Schema conversion errors are reported rather than
silently dropping fields. Its Markdown mode can participate in the parsing track.

Upstream installation/runtime instructions:
[Docling](https://docling-project.github.io/docling/installation/),
[Marker](https://github.com/datalab-to/marker),
[Unstructured](https://docs.unstructured.io/open-source/installation/full-installation),
[MinerU](https://github.com/opendatalab/MinerU),
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[olmOCR](https://github.com/allenai/olmocr),
[PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/),
[NuExtract](https://github.com/numindai/nuextract).

### Reader options and input files

`configs/readers.yaml` lists every reader. Each entry contains the options passed
to that tool. The same tool can be compared in separate runs with different configs.
For example:

```yaml
tools:
  docling:
    pipeline_options: {do_ocr: true, do_table_structure: true}
  marker:
    config: {use_llm: false}
  unstructured:
    partition: {strategy: hi_res, infer_table_structure: true}
  mineru: {backend: hybrid-engine}
  paddleocr: {variant: vl} # structure_v3 selects the separate modular pipeline
  pymupdf4llm:
    markdown: {use_ocr: false}
  nuextract: {model: numind/NuExtract3, max_tokens: 16384}
```

Hosted readers accept `base_url`, `timeout_s`, `request_timeout_s`, and
`poll_interval_s`, plus documented provider options. Local readers accept `timeout`
(seconds); Docling `pipeline_options`/`convert`, Marker `config`, Unstructured
`partition`, PaddleOCR `init`/`predict`, PyMuPDF4LLM `markdown`, and MinerU/olmOCR
`cli_args` (a list of individual arguments). Keep credentials in environment variables.
Pin provider model/tier versions in the config for comparable experiments.

The bundled `datasets/readers.jsonl` contains three small image fixtures for wiring
checks. These are not an accuracy benchmark; tools restricted to PDFs need a PDF
manifest. Replace it with your documents:

```json
{"doc_id":"annual-report-2025","doc_path":"reports/2025.pdf","suite":"financial-report"}
{"doc_id":"scanned-form","doc_path":"forms/scan.pdf","gt_path":"gold/scan.txt","suite":"scan"}
```

JSON arrays and JSONL are supported. Config paths resolve relative to the YAML
file; document and gold paths resolve relative to the manifest directory or an
explicit `root`. Text gold is optional in parsing mode. If supplied, the configured
text metrics run; they do not measure table structure, charts, or numeric fidelity.
Without gold, the run saves outputs without claiming an accuracy score.

### Omni-Extract-Bench

[Omni-Extract-Bench](https://www.datalab.to/blog/omni-extract-bench) measures
**schema-based structured extraction**. The integration reads its released Parquet
manifest, PDFs, JSON gold, and inline byte-encoded schemas, and calls the official
`omni-extract-bench==0.1.2` scorer. It preserves the original schema and the submitted
schema after the official reference/default normalization.

Use Python **3.11+** for this optional track:

```bash
pip install -e '.[omni]'
hf download datalab-to/omni_extract_bench --repo-type dataset \
  --revision 8e45d8ab5a8794296d2d5920cf87910cdc403527 \
  --local-dir datasets/omni_extract_bench
# This pins the release inspected on 2026-09-16; update deliberately for later releases.

docbench run-readers -c configs/omni_extract.yaml --list
docbench run-readers -c configs/omni_extract.yaml --limit 3
```

The config starts with three documents; `limit: null` evaluates the complete
manifest. No corpus download or paid request happens automatically on installation.
The downloaded dataset stays ignored by Git.

Datalab, Reducto, Mistral and NuExtract receive the schema directly. Parser-only
tools first produce text/Markdown and then use the **same shared extraction model**:
set `DOCBENCH_EXTRACT_BASE_URL` and `DOCBENCH_EXTRACT_MODEL`, plus
`DOCBENCH_EXTRACT_API_KEY` if that endpoint needs authentication. Empty shared settings
produce a configuration error for those enabled tools without calling their parsers;
native extractors can still run. These results are labelled `parse_then_extract` and
include the downstream model name. They measure the combined system.

The official scorer emits per-field verdicts for matches, misreads, missing values,
and fabricated/invented values. `accuracy` is **0–100**; `precision`, `recall`, and
`f1` are **0–1**. Scores are conditional on scored documents, with attempted,
succeeded, scored, and failure counts recorded. Empty/error extractions and
truncated or invalid JSON are failures, not successful zero-score predictions.
Dataset and scoring details: [dataset](https://huggingface.co/datasets/datalab-to/omni_extract_bench),
[official implementation](https://github.com/datalab-to/omni_extract_bench).

### Inspecting results

Each run creates a new `results/<run>/` directory with `results.json` and a linked
`report.md`. Existing run directories are never overwritten. Each document/reader
has its original response in `raw.json`, readable `output.md`, and `result.json`
with source path/hash, status, options at the tool level, and elapsed times.
Extraction adds `prediction.json`, `expected.json`, both schemas, and `verdicts.json`;
combined runs also retain `extraction_raw.json`. Raw parser output survives downstream
extraction/scoring failure. Failed calls do not stop other tools and produce a nonzero
CLI exit code. Disabled tools are listed separately.

Local calls start a fresh process for each document, so timing includes model startup.
These are cold document timings, not a warmed throughput benchmark. Official scoring
time is recorded separately from inference. Local native output includes library
versions where available; hosted responses retain provider metadata. No quality or
speed ranking is implied by a wiring smoke test.

## Capability scorecards

Every reader run now writes a `scorecard.json` per tool and a linked Markdown report.
Scores are **per document**, averaging successful repetitions before averaging documents.
For ParseBench, a document/group pair is an evaluation unit; the scorecard also reports
distinct source-document hashes and family counts separately.
There is no weighted overall winner: table structure, numeric fidelity, and missing
content remain separate. Every metric includes its document count and direction where
known. Category and tag groups can overlap; never sum their counts as unique documents.

| Rubric | Measurement |
|---|---|
| Text fidelity | Exact character/word edit rates (CER/WER) when transcription gold exists |
| Completeness | Missing, extra and duplicated token rates, explicitly labelled bag-of-token proxies; official ParseBench content rules |
| Numeric fidelity | Annotated field values, signs, units, currencies and dates; explicit tolerances and label context |
| Structure | Official ParseBench table/header, formatting and reading-order metrics; OmniDocBench table and order metrics |
| Charts | Official ParseBench labelled data-point checks with released tolerances |
| Traceability | Official ParseBench visual grounding when native geometry is available; optional page/box checks |
| Reliability | Inference completion, inference failures, scoring failures and unsupported scoring reported separately |
| Robustness | Category/tag slices and explicit paired `variant_of` deltas; no claim of robustness without annotated variants |
| Repeatability | Output-hash agreement and score variance across repetitions, with failed runs counted |
| Performance | Median/p95 latency, parsing and downstream time, actual-wall-time document/page throughput |
| Cost | Configured-rate USD estimates, known-cost coverage, cost per 1,000 attempted pages and usable document |
| Downstream usefulness | Optional fixed-model QA; existing fixed-model schema extraction; both labelled combined-system measurements |

An unavailable metric stays **not measured**, never zero. Partial official evaluations
retain the supported metrics and their unsupported-capability reason. A successful parse
with no native bounding boxes counts as successful inference but does not establish
visual-grounding accuracy. Raw output, gold/rules, normalized output, official verdicts,
and source links remain available in each document's artifact directory.

Use `repetitions`, `concurrency`, and `warmup` in any reader YAML:

```yaml
repetitions: 3
concurrency: 2
warmup: 1
costs: {}  # Unknown until you supply actual rates; an absent price is not free.
environment: {hardware: "describe CPU/GPU and memory here"}
```

Each price entry is keyed by reader name, or `shared_extractor` for downstream calls,
and accepts `per_document_usd`, `per_page_usd`, and `per_second_usd`. Rates add together;
an explicit zero can describe free inference. The report includes failed, repeated and
warmup attempts in its total-run cost section. Prices are estimates, not billing data;
unreported provider-internal retries and charges cannot be inferred. Page-based metrics
require an annotated page count or a readable PDF/image. Quality scoring is outside the
throughput timing window. Local reader processes still start afresh for every call, so
warmup does **not** produce warmed-model local latency. No memory or GPU utilization
measurement is claimed; record hardware explicitly for meaningful comparisons.

### ParseBench: five parsing capabilities

The [official dataset](https://huggingface.co/datasets/llamaindex/ParseBench) has
`table`, `chart`, `text_content`, `text_formatting`, and `layout` groups. OCR is an
`ocr` tag within text groups. Other tags include `multicolumns`, `multilang`, difficulty,
and handwriting. Documents are parsed once per document/group, never once per rule.

Use Python **3.12+** for the optional `parse-bench==1.0.4` evaluator. To isolate it,
install the package in another environment and set `PARSEBENCH_PYTHON` in `.env` to
that environment's Python. Blank uses the benchmark's interpreter.

```bash
pip install -e '.[parsebench,omni]'
hf download llamaindex/ParseBench --repo-type dataset \
  --revision 2805a1d940f95a203e0ae4b88be9934f7765b3fc \
  --local-dir datasets/parsebench
docbench run-readers -c configs/parsebench.yaml --inventory
docbench run-readers -c configs/parsebench.yaml --list
docbench run-readers -c configs/parsebench.yaml
docbench run-readers -c configs/parsebench.yaml --category text_content --tag ocr
```

The example selects at most three documents per category; set `per_category_limit: null`
for the full dataset. For a partial corpus, list the downloaded categories explicitly
in `benchmark_options.categories`; missing requested category files fail before inference.
Native layout normalization uses upstream provider adapters. Some adapters need optional
provider packages such as `parse-bench[llamaparse,docling,reducto,datalab]` in the evaluator
environment. Missing geometry or its adapter is visible as unsupported. The optional
paid LLM chart normalization is always disabled. Official metrics retain their names
and scales, including diagnostic counts. Our document averages are not advertised as
the upstream leaderboard's aggregate. [Official evaluator](https://github.com/run-llama/ParseBench)

### OmniDocBench: text, tables, formulas and reading order

Use the [official checkout](https://github.com/opendatalab/OmniDocBench) at
`f133a71e9e91c3621c7ce8994200a7b394a06eb3` in a separate Python **3.11** environment
(this upstream version requires Python below 3.12). Install it with `pip install -e`
pointing to that checkout. Set `OMNIDOCBENCH_ROOT` and `OMNIDOCBENCH_PYTHON` in `.env`.

```bash
hf download opendatalab/OmniDocBench --repo-type dataset \
  --revision aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec \
  --include OmniDocBench.json 'images/*' --local-dir datasets/omnidocbench
docbench run-readers -c configs/omnidocbench.yaml --inventory
docbench run-readers -c configs/omnidocbench.yaml
```

For a smaller check, point the config's `manifest` to
`<checkout>/demo_data/omnidocbench_demo/OmniDocBench_demo.json` and `root` to its parent.
This integration executes the official Markdown end-to-end evaluator. It reports text,
formula and reading-order edit distances (lower is better), table TEDS/structure TEDS
(higher is better), and optional formula CDM. Enable `benchmark_options.cdm: true` only
after installing the upstream rendering prerequisites: TeX Live/CJK, Ghostscript and
ImageMagick 7 with PDF support. Missing CDM dependencies remain unavailable. Chart-value
extraction and bounding-box detection are not measured by this particular track.
Page/block attributes become filterable tags such as `language:english`.

### Your financial corpus and held-out evaluation

`configs/financial_pilot.yaml` demonstrates annotations against the existing synthetic
Acme text fixture. It is a wiring example, **not** a real financial benchmark. Replace
its manifest with your reviewed documents. Use multiple categories/tags per document:

```json
{"doc_id":"report-2025","doc_path":"reports/2025.pdf","categories":["table","chart","text_content"],"tags":["financial","annual_report","ocr"],"family_id":"issuer-report-template","split":"test","page_count":80}
```

An optional `annotations` JSONL sidecar joins on `doc_id`, so an upstream corpus need
not be modified. It can add `categories`, `tags`, `family_id`, `split`, `page_count`,
`variant_of`, `text_gold`, `checks`, `qa`, and `annotation_note`. Sidecar text-gold paths
are relative to the sidecar; manifest paths are relative to `root`. Unknown/duplicate
annotation IDs fail rather than silently dropping labels. Category filters match any
selected category; tag filters require all selected tags. Family IDs spanning multiple
declared splits are rejected before filtering; identical source bytes across declared
splits are also rejected. Group related pages, filings and templates into explicit
families: source-path grouping cannot discover every common template automatically.

```bash
docbench run-readers -c configs/financial_pilot.yaml --inventory
docbench run-readers -c configs/financial_pilot.yaml --repetitions 3 --concurrency 2
# Once a reviewed corpus contains those labels:
docbench run-readers -c configs/financial_pilot.yaml --split test --tag financial
```

See `datasets/financial_annotations.jsonl` for working numeric, currency, period, reading
order and QA examples. Numeric checks require a field context or a regex with a named
`value` group; ambiguous multiple values remain unmeasured unless the annotation selects
one. Values are not silently rescaled: annotate the expected displayed number and unit.
Optional `location` checks require normalized `result.elements` with one-based page and
normalized `[x0,y0,x1,y1]` boxes; the current readers preserve raw native geometry, while
ParseBench performs provider-specific normalization for its own layout evaluator.

To compare robustness, supply a reviewed altered document with `variant_of` set to its
baseline `doc_id`, and keep both in the same family/split. Paired deltas use only shared
metrics and show failures and missing pairs. No synthetic degradation is generated
automatically. Keep dev and test families separate when tuning prompts/options.

For downstream QA, set `downstream_qa: true` and the existing `DOCBENCH_EXTRACT_*`
variables. Every reader uses that same model, prompt and question schema. Gold answers
are kept out of the model request and used only for scoring; raw answers and individual
verdicts are retained. These QA scores describe the parser-plus-answerer system.

Omni-Extract-Bench's four `suite` values describe data sources, not table/chart/OCR
capabilities. Add reviewed sidecar tags for those slices; unlabelled data remains visible.

## What the existing QA/stress commands measure

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
