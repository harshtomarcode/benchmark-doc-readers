"""Optional document readers, run in their own installed Python environments.

No reader dependency is imported in the benchmark process. Environment variables
select existing executables; this module never installs packages or starts a
NuExtract model server. Options are passed to the documented APIs below.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# A separate result file keeps model download/progress logging out of the JSON.
# Official APIs: docling-project.github.io/docling/reference/document_converter/
# github.com/datalab-to/marker, docs.unstructured.io/open-source/core-functionality/partitioning
# github.com/PaddlePaddle/PaddleOCR, pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html
# github.com/numindai/nuextract (NuExtract3 template and multi-page PDF examples).
_WORKER = r"""
import base64
import importlib.metadata
import io
import json
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name, path, options = request["name"], request["path"], request["options"]
text, structured, raw, error = "", None, None, None
packages = []

if name == "docling":
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    settings = PdfPipelineOptions(**options.get("pipeline_options", {}))
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=settings)
    })
    result = converter.convert(path, **options.get("convert", {}))
    text = result.document.export_to_markdown()
    structured = result.document.export_to_dict()
    raw = {"document": structured, "status": result.status,
           "errors": result.errors, "pipeline_options": settings.model_dump()}
    if getattr(result.status, "value", str(result.status)) != "success":
        error = "Docling conversion did not fully succeed: " + str(result.status)
    packages = ["docling", "docling-core"]

elif name == "marker":
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict
    from marker.renderers.markdown import MarkdownRenderer
    from marker.renderers.json import JSONRenderer
    config = dict(options.get("config", {}))
    config.setdefault("paginate_output", True)
    config.setdefault("html_tables_in_markdown", True)
    parser = ConfigParser(config)
    converter = PdfConverter(config=parser.generate_config_dict(),
        artifact_dict=create_model_dict(), processor_list=parser.get_processors(),
        llm_service=parser.get_llm_service())
    # Run inference once; render the same document both ways to retain boxes.
    document = converter.build_document(path)
    markdown = converter.resolve_dependencies(MarkdownRenderer)(document)
    blocks = converter.resolve_dependencies(JSONRenderer)(document)
    text = markdown.markdown
    structured = blocks.model_dump()
    raw = {"markdown": markdown.model_dump(), "blocks": structured,
           "config": converter.config}
    packages = ["marker-pdf"]

elif name == "unstructured":
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        from unstructured.partition.pdf import partition_pdf as partition
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".heic", ".webp"):
        from unstructured.partition.image import partition_image as partition
    else:
        from unstructured.partition.auto import partition
    kwargs = {"strategy": "hi_res", "infer_table_structure": True,
              "include_page_breaks": True} if suffix in (
                  ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".heic", ".webp"
              ) else {}
    kwargs.update(options.get("partition", {}))
    elements = partition(filename=path, **kwargs)
    structured = [element.to_dict() for element in elements]
    # Keep native table HTML when available, rather than flattening its cells.
    text = "\n\n".join(element.get("metadata", {}).get("text_as_html")
                        or element.get("text", "") for element in structured)
    raw = {"elements": structured, "partition_options": kwargs}
    packages = ["unstructured", "unstructured-inference"]

elif name == "paddleocr":
    from paddleocr import PaddleOCRVL, PPStructureV3
    variant = options.get("variant", "vl")
    if variant not in ("vl", "structure_v3"):
        raise ValueError("paddleocr variant must be vl or structure_v3")
    kwargs = dict(options.get("init", {}))
    if variant == "vl":
        kwargs.setdefault("pipeline_version", "v1.6")
    pipeline = (PaddleOCRVL if variant == "vl" else PPStructureV3)(**kwargs)
    pages = list(pipeline.predict(input=path, **options.get("predict", {})))
    original_pages = [page.json for page in pages]
    if options.get("restructure") is not None:
        if variant != "vl":
            raise ValueError("restructure is supported only for paddleocr variant vl")
        pages = pipeline.restructure_pages(pages, **options["restructure"])
    markdown_pages = [page.markdown for page in pages]
    text = pipeline.concatenate_markdown_pages(markdown_pages)
    structured = [page.json for page in pages]
    raw = {"pages": original_pages, "result_pages": structured,
           "markdown_pages": markdown_pages, "variant": variant, "init": kwargs}
    packages = ["paddleocr", "paddlex", "paddlepaddle", "paddlepaddle-gpu"]

elif name == "pymupdf4llm":
    import pymupdf4llm
    kwargs = dict(options.get("markdown", {}))
    kwargs["page_chunks"] = True
    kwargs["write_images"] = False
    kwargs.setdefault("embed_images", True)
    structured = pymupdf4llm.to_markdown(path, **kwargs)
    text = "\n\n".join(page["text"] for page in structured)
    raw = {"pages": structured, "markdown_options": kwargs}
    packages = ["pymupdf4llm", "PyMuPDF"]

elif name == "nuextract":
    # NuExtract consumes a typed output template, not JSON Schema.
    template = options.get("template")
    descriptions = None
    if template is None and request.get("schema") is not None:
        from numind.nuextract_utils import convert_json_schema_to_nuextract_template
        template, dropped, descriptions = convert_json_schema_to_nuextract_template(
            request["schema"])
        if dropped:
            raise ValueError("NuExtract cannot represent schema branches: "
                             + json.dumps(dropped) + "; supply options.template explicitly")
        packages.append("numind")
    def validate_template(value, location="$template"):
        types = {"string", "verbatim-string", "number", "integer", "boolean", "date",
                 "time", "date-time", "duration", "country", "currency", "language",
                 "language-tag", "script", "url", "email-address", "phone-number",
                 "iban", "bic", "unit-code"}
        if isinstance(value, dict) and value:
            for key, child in value.items():
                validate_template(child, location + "." + key)
        elif isinstance(value, list) and value:
            if len(value) == 1 and isinstance(value[0], (dict, list)):
                validate_template(value[0], location + "[]")
            elif not all(isinstance(item, str) for item in value):
                raise ValueError("Unsupported NuExtract array/enum at " + location)
        elif isinstance(value, str) and (value in types or value.startswith("region:")):
            pass
        else:
            raise ValueError("Invalid NuExtract template type at " + location
                             + "; use typed leaves, not JSON Schema")
    if template is not None:
        validate_template(template)
    content = []
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            for page in document:
                pixels = page.get_pixmap(dpi=int(options.get("dpi", 170)), alpha=False)
                data = base64.b64encode(pixels.tobytes("png")).decode("ascii")
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + data}})
        packages.append("PyMuPDF")
    elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
        mime = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp"}[suffix]
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]
    elif suffix in (".txt", ".md", ".html", ".csv", ".json"):
        content = [{"type": "text", "text": Path(path).read_text(encoding="utf-8")}]
    else:
        raise ValueError("NuExtract accepts PDF, PNG/JPEG/WebP or UTF-8 text documents")
    if not content:
        raise ValueError("NuExtract input has no pages")
    structured = {"template": template, "descriptions": descriptions, "content": content}
    raw = {}
else:
    raise ValueError("Unknown worker reader: " + name)

versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass

def serialize(value):
    # Preserve native arrays and image assets instead of lossy default=str.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "save") and hasattr(value, "size"):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return {"mime_type": "image/png",
                "base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    if hasattr(value, "value"):
        return value.value
    raise TypeError("Cannot serialize native reader value " + type(value).__name__)

Path(sys.argv[2]).write_text(json.dumps({"text": text, "structured": structured,
    "raw": {"output": raw, "versions": versions}, "error": error},
    default=serialize, ensure_ascii=False),
    encoding="utf-8")
"""


def _run_process(command: list[str], directory: Path, timeout: float) -> dict:
    """Run an isolated parser and stop its model-server descendants on timeout."""
    process = subprocess.Popen(
        command,
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise TimeoutError(f"Reader process exceeded {timeout:g} seconds") from exc
        raise
    if process.returncode:
        raise RuntimeError(
            f"Reader process exited {process.returncode}: {(stderr or stdout)[-12000:]}"
        )
    return {"stdout": stdout, "stderr": stderr}


def read_local(name: str, path: Path, *, options: dict, schema: dict | None = None) -> dict:
    """Read the actual input document; return text, structured data and native output.

    Gates: <NAME>_PYTHON for Python packages, MINERU_COMMAND for its CLI,
    NUEXTRACT_BASE_URL for an already-running NuExtract3-compatible endpoint.
    NUEXTRACT_PYTHON optionally selects the PDF/schema preparation interpreter.
    Every invocation loads a fresh process; elapsed time includes model startup.
    """
    supported = {
        "docling",
        "marker",
        "unstructured",
        "mineru",
        "paddleocr",
        "olmocr",
        "pymupdf4llm",
        "nuextract",
    }
    if name not in supported:
        raise ValueError(f"Unknown local reader: {name}")
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Not a document file: {path}")
    options = dict(options)
    timeout = float(options.get("timeout", 600))
    if timeout <= 0:
        raise ValueError("Reader timeout must be positive")
    if name == "nuextract" and not os.environ.get("NUEXTRACT_BASE_URL"):
        raise ValueError("Set NUEXTRACT_BASE_URL to an existing NuExtract3 endpoint")
    variable = "MINERU_COMMAND" if name == "mineru" else name.upper() + "_PYTHON"
    executable = os.environ.get(variable)
    if name == "nuextract" and not executable:
        executable = sys.executable
    if not executable:
        raise ValueError(f"Set {variable} to the reader's installed executable")
    executable = shutil.which(os.path.expanduser(executable))
    if not executable:
        raise ValueError(f"{variable} does not identify an executable file or command")

    with tempfile.TemporaryDirectory(prefix=f"docbench-{name}-") as temporary:
        directory = Path(temporary)
        if name in {"mineru", "olmocr"}:
            destination = directory / "output"
            extra = options.get("cli_args", [])
            if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
                raise ValueError("cli_args must be a list of individual argument strings")
            if name == "mineru":
                version = _run_process([executable, "--version"], directory, timeout)["stdout"]
                command = [
                    executable,
                    "-p",
                    str(path),
                    "-o",
                    str(destination),
                    "-b",
                    options.get("backend", "hybrid-engine"),
                ]
                for key, flag in (("method", "-m"), ("language", "-l"), ("effort", "--effort")):
                    if key in options:
                        command.extend([flag, str(options[key])])
            else:
                version = _run_process(
                    [
                        executable,
                        "-c",
                        "from importlib.metadata import version; print(version('olmocr'))",
                    ],
                    directory,
                    timeout,
                )["stdout"]
                command = [
                    executable,
                    "-m",
                    "olmocr.pipeline",
                    str(destination),
                    "--markdown",
                    "--pdfs",
                    str(path),
                ]
                for key, flag in (("model", "--model"), ("server", "--server")):
                    if key in options:
                        command.extend([flag, str(options[key])])
            logs = _run_process(command + extra, directory, timeout)
            files = {}
            for output in sorted(destination.rglob("*")):
                if not output.is_file():
                    continue
                relative = str(output.relative_to(destination))
                if output.suffix == ".json":
                    files[relative] = json.loads(output.read_text(encoding="utf-8"))
                elif output.suffix == ".jsonl":
                    files[relative] = [
                        json.loads(line)
                        for line in output.read_text(encoding="utf-8").splitlines()
                        if line
                    ]
                elif output.suffix in {".md", ".html"}:
                    files[relative] = output.read_text(encoding="utf-8")
                elif output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    files[relative] = {
                        "base64": base64.b64encode(output.read_bytes()).decode("ascii")
                    }
            markdown = [value for key, value in files.items() if key.endswith(".md")]
            if len(markdown) != 1:
                raise RuntimeError(
                    f"{name} produced {len(markdown)} Markdown documents; expected 1"
                )
            structured = {
                key: value for key, value in files.items() if key.endswith((".json", ".jsonl"))
            }
            return {
                "text": markdown[0],
                "structured": structured or None,
                "raw": {"files": files, "version": version.strip(), **logs},
            }

        request_path, response_path = directory / "request.json", directory / "response.json"
        request_path.write_text(
            json.dumps({"name": name, "path": str(path), "options": options, "schema": schema}),
            encoding="utf-8",
        )
        logs = _run_process(
            [executable, "-c", _WORKER, str(request_path), str(response_path)], directory, timeout
        )
        result = json.loads(response_path.read_text(encoding="utf-8"))
        if name != "nuextract":
            result["raw"].update(logs)
            return result

        import httpx

        prepared = result["structured"]
        template_options = {"enable_thinking": bool(options.get("enable_thinking", False))}
        if prepared["template"] is not None:
            template_options["template"] = json.dumps(prepared["template"], ensure_ascii=False)
        else:
            template_options["mode"] = "markdown"
        instructions = options.get("instructions", "")
        if prepared["descriptions"]:
            instructions += "\nField descriptions: " + json.dumps(prepared["descriptions"])
        if instructions:
            template_options["instructions"] = instructions
        body = {
            "model": options.get("model", "numind/NuExtract3"),
            "messages": [{"role": "user", "content": prepared["content"]}],
            "temperature": options.get("temperature", 0),
            "max_tokens": int(options.get("max_tokens", 16384)),
            "chat_template_kwargs": template_options,
        }
        base_url = os.environ["NUEXTRACT_BASE_URL"].rstrip("/")
        key = os.environ.get(options.get("api_key_env", "NUEXTRACT_API_KEY"), "").strip()
        response = httpx.post(
            base_url + "/chat/completions",
            json=body,
            headers={"Authorization": "Bearer " + key} if key else {},
            timeout=timeout,
        )
        try:
            original = response.json()
        except ValueError:
            original = {"response_text": response.text}
        result = {
            "text": "",
            "structured": None,
            "raw": {
                "response": original,
                "template": prepared["template"],
                "descriptions": prepared["descriptions"],
                "input_parts": len(prepared["content"]),
                "versions": result["raw"]["versions"],
                **logs,
            },
        }
        if response.is_error:
            result["error"] = f"NuExtract endpoint returned HTTP {response.status_code}"
            return result
        try:
            choice = original["choices"][0]
            text = choice["message"].get("content")
            if not isinstance(text, str):
                raise TypeError("NuExtract returned no textual content")
            result["text"] = text
            if choice.get("finish_reason") == "length":
                raise ValueError("NuExtract hit max_tokens; output is incomplete")
            answer = text.split("</think>")[-1].strip()
            if prepared["template"] is not None:
                if answer.startswith("```") and answer.endswith("```"):
                    answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                result["structured"] = json.loads(answer)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            result["error"] = f"Invalid NuExtract output: {exc}"
        return result
