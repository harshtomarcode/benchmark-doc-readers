"""Hosted document readers, using their documented REST APIs.

``options`` accepts base_url, timeout_s (whole job, default 600),
request_timeout_s (default 120), and poll_interval_s (default 2). Other keys
are sent as provider-specific request options. Credentials come only from env.
Unstructured uses its legacy Partition API, not the Workflow platform API.
"""

from __future__ import annotations

import copy
import json
import math
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx


def read_cloud(
    name: str, path: Path, *, options: dict, schema: dict | None = None
) -> dict:
    """Upload one document and return text, optional extracted data, and raw output.

    Supplying schema calls native schema extraction for Datalab, Reducto, or
    Mistral. LlamaParse and Unstructured explicitly reject it before uploading.
    ``raw`` contains provider responses, never request headers or credentials.
    No automatic submission retries: retrying can create another billable job.
    """
    providers = {
        "llamaparse": ("LLAMA_CLOUD_API_KEY", "https://api.cloud.llamaindex.ai"),
        "datalab": ("DATALAB_API_KEY", "https://www.datalab.to"),
        "reducto": ("REDUCTO_API_KEY", "https://platform.reducto.ai"),
        "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai"),
        "unstructured_api": ("UNSTRUCTURED_API_KEY", "https://api.unstructuredapp.io"),
    }
    if name not in providers:
        raise ValueError(f"Unknown cloud reader: {name}")
    if schema is not None and name in {"llamaparse", "unstructured_api"}:
        raise NotImplementedError(
            f"{name} reader does not implement native schema extraction; "
            "parse first and use a configured downstream extractor"
        )
    env_name, default_base = providers[name]
    api_key = os.environ.get(env_name)
    if not api_key:
        raise ValueError(f"Set {env_name} to use {name}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = copy.deepcopy(options)
    base = str(payload.pop("base_url", default_base)).rstrip("/")
    timeout = float(payload.pop("timeout_s", 600))
    request_timeout = float(payload.pop("request_timeout_s", 120))
    interval = float(payload.pop("poll_interval_s", 2))
    if any(
        not math.isfinite(value) or value <= 0
        for value in (timeout, request_timeout, interval)
    ):
        raise ValueError("Cloud timeout and polling options must be positive, finite seconds")
    deadline = time.monotonic() + timeout
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if name == "datalab":
        headers = {"X-API-Key": api_key}
    elif name == "unstructured_api":
        headers = {"unstructured-api-key": api_key}
    else:
        headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(follow_redirects=False) as client:

        def failed(raw: Any, message: str, text: str = "") -> dict:
            return {"text": text, "structured": None, "raw": raw, "error": message}

        def request(method: str, url: str, *, authenticated: bool = True, **kwargs: Any) -> Any:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{name} exceeded timeout_s={timeout:g}")
            # Result URLs can point at object storage. Never send API keys there.
            if authenticated and urlsplit(url).netloc != urlsplit(base).netloc:
                raise ValueError(f"{name} returned a polling URL outside its API host")
            try:
                response = client.request(
                    method, url, headers=headers if authenticated else {},
                    timeout=min(request_timeout, remaining), **kwargs,
                )
            except httpx.TimeoutException:
                raise TimeoutError(f"{name} HTTP request timed out") from None
            except httpx.HTTPError:
                raise RuntimeError(f"{name} HTTP transport failed") from None
            if not response.is_success:
                # Do not persist server echoes of credentials or signed URL queries.
                raise RuntimeError(f"{name} returned HTTP {response.status_code}")
            try:
                return response.json()
            except ValueError:
                raise ValueError(f"{name} returned a non-JSON response") from None

        def poll(
            url: str, *, nested_job: bool = False, **kwargs: Any
        ) -> tuple[Any, str | None]:
            result = None
            while True:
                try:
                    result = request("GET", url, **kwargs)
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    if result is None:
                        raise
                    return result, str(exc)
                if not isinstance(result, dict):
                    return result, f"{name} returned a non-object polling response"
                job = result.get("job", {}) if nested_job else result
                if not isinstance(job, dict):
                    return result, f"{name} polling response has no job object"
                status = str(job.get("status") or "").lower()
                if job.get("success") is False or status in {
                    "failed", "error", "cancelled", "canceled", "aborted",
                }:
                    return result, f"{name} document processing failed ({status or 'error'})"
                if status in {"complete", "completed", "success"}:
                    return result, None
                if not status:
                    return result, f"{name} polling response has no job status"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return result, f"{name} exceeded timeout_s={timeout:g}"
                time.sleep(min(interval, remaining))

        if name == "llamaparse":
            # https://developers.llamaindex.ai/llamaparse/parse/guides/api-reference/
            payload = {"tier": "agentic", "version": "latest", **payload}
            with path.open("rb") as file:
                submitted = request(
                    "POST", f"{base}/api/v2/parse/upload",
                    files={"file": (path.name, file, mime)},
                    data={"configuration": json.dumps(payload)},
                )
            job = submitted.get("job", submitted) if isinstance(submitted, dict) else None
            if not isinstance(job, dict) or not job.get("id"):
                return failed(submitted, "llamaparse submission returned no job ID")
            job_id = quote(str(job["id"]), safe="")
            raw, error = poll(
                f"{base}/api/v2/parse/{job_id}", nested_job=True,
                params={"expand": "markdown_full,items,metadata,usage"},
            )
            if error:
                return failed(raw, error)
            text = raw.get("markdown_full")
            if not isinstance(text, str):
                return failed(raw, "llamaparse completed without markdown_full text output")
            return {"text": text, "structured": None, "raw": raw}

        if name == "datalab":
            # https://documentation.datalab.to/docs/recipes/conversion/conversion-api-overview
            # https://documentation.datalab.to/docs/recipes/structured-extraction/api-overview
            payload = {"mode": "balanced", "output_format": "markdown", **payload}
            if schema is not None:
                payload["page_schema"] = json.dumps(schema)
                payload.setdefault("extraction_mode", "balanced")
            endpoint = "extract" if schema is not None else "convert"
            form = {
                key: json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
                for key, value in payload.items() if value is not None
            }
            with path.open("rb") as file:
                submitted = request(
                    "POST", f"{base}/api/v1/{endpoint}", data=form,
                    files={"file": (path.name, file, mime)},
                )
            if not isinstance(submitted, dict) or submitted.get("success") is False:
                return failed(submitted, "datalab rejected the document processing request")
            if not isinstance(submitted.get("request_check_url"), str):
                return failed(submitted, "datalab submission returned no polling URL")
            raw, error = poll(urljoin(base + "/", submitted["request_check_url"]))
            if error:
                return failed(raw, error)
            result = raw
            if raw.get("result_url"):
                try:
                    downloaded = request("GET", raw["result_url"], authenticated=False)
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    return failed(raw, str(exc))
                raw = {"response": raw, "downloaded_result": downloaded}
                if not isinstance(downloaded, dict):
                    return failed(raw, "datalab downloaded a non-object result")
                # Downloaded content is authoritative; polling stubs may contain empty fields.
                result = {**result, **downloaded}
                if result.get("success") is False or str(result.get("status")).lower() == "failed":
                    return failed(raw, "datalab downloaded result reports processing failure")
            text = result.get("markdown")
            if not isinstance(text, str):
                text = result.get("html")
            if not isinstance(text, str):
                if schema is None:
                    return failed(raw, "datalab completed without markdown or HTML text output")
                text = ""  # Turbo extraction legitimately omits parsed document output.
            structured = None
            error = None
            if schema is not None:
                structured = result.get("extraction_schema_json")
                if isinstance(structured, str):
                    try:
                        structured = json.loads(structured)
                    except json.JSONDecodeError:
                        structured = None
                        error = "datalab returned invalid JSON in extraction_schema_json"
                if structured is None:
                    error = error or "datalab completed without the requested extraction"
            if error:
                return failed(raw, error, text)
            return {"text": text, "structured": structured, "raw": raw}

        if name == "reducto":
            # https://docs.reducto.ai/parse/overview and /extract/overview
            # Async result envelope: reducto-python-sdk/types/job_get_response.py
            with path.open("rb") as file:
                uploaded = request(
                    "POST", f"{base}/upload", files={"file": (path.name, file, mime)},
                )
            if not isinstance(uploaded, dict) or not uploaded.get("file_id"):
                return failed(uploaded, "reducto upload returned no file ID")
            payload["input"] = uploaded["file_id"]
            if schema is not None:
                payload["instructions"] = {**payload.get("instructions", {}), "schema": schema}
            endpoint = "extract_async" if schema is not None else "parse_async"
            submitted = request("POST", f"{base}/{endpoint}", json=payload)
            if not isinstance(submitted, dict) or not submitted.get("job_id"):
                return failed(submitted, "reducto submission returned no job ID")
            job_id = quote(str(submitted["job_id"]), safe="")
            raw, error = poll(f"{base}/job/{job_id}")
            if error:
                return failed(raw, error)
            response = raw.get("result")
            if not isinstance(response, dict) or "result" not in response:
                return failed(raw, "reducto completed without its nested result response")
            result = response["result"]
            url_result = schema is None or payload.get("settings", {}).get("force_url_result")
            if url_result and isinstance(result, dict) and result.get("type") == "url":
                if not isinstance(result.get("url"), str):
                    return failed(raw, "reducto returned a URL result without a download URL")
                try:
                    result = request("GET", result["url"], authenticated=False)
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    return failed(raw, str(exc))
                raw = {"response": raw, "downloaded_result": result}
            if schema is not None:
                # Extract wraps a single document's object in a one-item list.
                if schema.get("type") == "object" and isinstance(result, list) and len(result) == 1:
                    result = result[0]
                if result is None:
                    return failed(raw, "reducto completed without the requested extraction")
                return {"text": "", "structured": result, "raw": raw}
            chunks = result.get("chunks") if isinstance(result, dict) else result
            if not isinstance(chunks, list) or not chunks:
                return failed(raw, "reducto completed without parsed chunks")
            if any(
                not isinstance(c, dict) or not isinstance(c.get("content"), str) for c in chunks
            ):
                return failed(raw, "reducto returned malformed parsed chunks")
            text = "\n\n".join(chunk["content"] for chunk in chunks)
            return {"text": text, "structured": None, "raw": raw}

        if name == "mistral":
            # https://docs.mistral.ai/api/endpoint/files and /api/endpoint/ocr
            # https://docs.mistral.ai/studio/document-processing/annotations
            with path.open("rb") as file:
                uploaded = request(
                    "POST", f"{base}/v1/files", data={"purpose": "ocr"},
                    files={"file": (path.name, file, mime)},
                )
            if not isinstance(uploaded, dict) or not uploaded.get("id"):
                return failed(uploaded, "mistral upload returned no file ID")
            file_id = quote(str(uploaded["id"]), safe="")
            signed = request("GET", f"{base}/v1/files/{file_id}/url")
            if not isinstance(signed, dict) or not isinstance(signed.get("url"), str):
                return failed(signed, "mistral returned no document URL")
            payload = {"model": "mistral-ocr-latest", **payload}
            if mime.startswith("image/"):
                payload["document"] = {"type": "image_url", "image_url": signed["url"]}
            else:
                payload["document"] = {"type": "document_url", "document_url": signed["url"]}
            if schema is not None:
                payload["document_annotation_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "document_extraction", "schema": schema, "strict": True,
                    },
                }
            raw = request("POST", f"{base}/v1/ocr", json=payload)
            if not isinstance(raw, dict) or not isinstance(raw.get("pages"), list):
                return failed(raw, "mistral returned no OCR page list")
            if not raw["pages"]:
                return failed(raw, "mistral returned an empty OCR page list")
            page_texts = []
            for page in raw["pages"]:
                if not isinstance(page, dict) or not isinstance(page.get("markdown"), str):
                    return failed(raw, "mistral returned an OCR page without markdown text")
                markdown = page["markdown"]
                # Explicit table_format returns table bodies separately from markdown.
                # https://docs.mistral.ai/studio/document-processing/basic_ocr
                tables = page.get("tables")
                if tables is None:
                    tables = []
                if not isinstance(tables, list) or any(
                    not isinstance(t, dict) or not isinstance(t.get("id"), str)
                    or not isinstance(t.get("content"), str) for t in tables
                ):
                    return failed(raw, "mistral returned malformed OCR tables")
                for table in tables:
                    table_id = table["id"]
                    markdown = markdown.replace(f"[{table_id}]({table_id})", table["content"])
                if any(
                    page.get(field) is not None and not isinstance(page[field], str)
                    for field in ("header", "footer")
                ):
                    return failed(raw, "mistral returned a non-text header or footer")
                page_texts.append("\n\n".join(
                    value for value in (page.get("header"), markdown, page.get("footer")) if value
                ))
            text = "\n\n".join(page_texts)
            structured = raw.get("document_annotation") if schema is not None else None
            error = None
            if schema is not None and structured is None:
                error = "mistral completed without the requested document annotation"
            if isinstance(structured, str):
                try:
                    structured = json.loads(structured)
                except json.JSONDecodeError:
                    structured = None
                    error = "mistral returned invalid JSON in document_annotation"
            if schema is not None and structured is None:
                error = error or "mistral completed without the requested document annotation"
            if error:
                return failed(raw, error, text)
            return {"text": text, "structured": structured, "raw": raw}

        # https://docs.unstructured.io/api-reference/legacy-api/partition/examples
        endpoint = payload.pop("endpoint_url", os.environ.get("UNSTRUCTURED_API_URL"))
        endpoint = endpoint or f"{base}/general/v0/general"
        # The documented account-specific URL may have a different hostname.
        base = endpoint
        payload = {"strategy": "auto", "output_format": "application/json", **payload}
        form = {
            key: json.dumps(value) if isinstance(value, (dict, bool)) else value
            for key, value in payload.items() if value is not None
        }
        with path.open("rb") as file:
            raw = request(
                "POST", endpoint, data=form, files={"files": (path.name, file, mime)},
            )
        if not isinstance(raw, list) or not raw:
            return failed(raw, "unstructured_api returned no partition elements")
        for element in raw:
            if not isinstance(element, dict):
                return failed(raw, "unstructured_api returned a malformed partition element")
            metadata = element.get("metadata")
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                return failed(raw, "unstructured_api returned malformed element metadata")
            value = metadata.get("text_as_html") or element.get("text")
            if not isinstance(value, str):
                return failed(raw, "unstructured_api returned an element without text")
        text = "\n\n".join(
            (element.get("metadata") or {}).get("text_as_html") or element.get("text", "")
            for element in raw
        )
        return {"text": text, "structured": None, "raw": raw}
