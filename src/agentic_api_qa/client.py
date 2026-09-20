"""Bounded, same-origin HTTP tool that returns sanitized evidence."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from agentic_api_qa.models import (
    Actor,
    EvidenceRecord,
    EvidenceRequest,
    EvidenceResponse,
    RequestSpec,
    RunConfig,
)


SENSITIVE_KEYS = {"authorization", "cookie", "password", "secret", "token", "x-api-key"}
SELECTED_RESPONSE_HEADERS = {"content-type", "retry-after", "x-request-id", "x-trace-id"}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _normalize_json_body(value: Any) -> Any:
    """Decode a json_body that arrived as a JSON string.

    An agent routinely emits `json_body` as the *text* of a JSON document
    rather than as a structure. Handing that string to httpx's ``json=``
    encodes it a second time, so the target receives a JSON string where an
    object was intended — which reads downstream as a malformed-input finding
    against the target instead of a defect in this harness.
    """
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
        if isinstance(decoded, (dict, list)):
            return decoded
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _excerpt(value: str, limit: int) -> str:
    return value.encode("utf-8", errors="replace")[:limit].decode(
        "utf-8", errors="replace"
    )


def _same_origin_url(base_url: str, path: str) -> str:
    base = urlparse(base_url)
    url = urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))
    parsed = urlparse(url)
    if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
        raise ValueError("Cross-origin HTTP tool calls are not allowed")
    return url


async def execute_request(
    *,
    config: RunConfig,
    request: RequestSpec,
    actor: Actor,
    iteration: int,
    scenario_or_rule_id: str,
) -> EvidenceRecord:
    url = _same_origin_url(str(config.base_url), request.path)
    redacted_headers = {
        name: "[REDACTED]" if name.lower() in config.redacted_headers else value
        for name, value in request.headers.items()
    }
    json_body = _normalize_json_body(request.json_body)
    request_body = (
        request.raw_body
        if request.raw_body is not None
        else _serialize(_sanitize(json_body))
    )
    evidence_request = EvidenceRequest(
        method=request.method,
        url=url,
        redacted_headers=redacted_headers,
        body_sha256=_digest(request_body),
        sanitized_body_excerpt=_excerpt(
            request_body, config.limits.max_response_excerpt_bytes
        ),
    )

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=config.limits.request_timeout_seconds,
            follow_redirects=True,
        ) as client:
            kwargs: dict[str, Any] = {
                "headers": request.headers,
                "params": request.query,
            }
            if request.raw_body is not None:
                kwargs["content"] = request.raw_body
            elif json_body is not None:
                kwargs["json"] = json_body
            response = await client.request(request.method, url, **kwargs)
        duration_ms = (time.perf_counter() - started) * 1_000
        content_type = response.headers.get("content-type", "")
        try:
            parsed_body: Any = response.json()
            json_parseable: bool | None = True
        except (json.JSONDecodeError, ValueError):
            parsed_body = response.text
            json_parseable = False if "json" in content_type.lower() else None
        sanitized_body = _serialize(_sanitize(parsed_body))
        encoded_body = sanitized_body.encode("utf-8", errors="replace")
        if isinstance(parsed_body, dict):
            json_type = "object"
            json_keys = sorted(str(key) for key in parsed_body.keys())
        elif isinstance(parsed_body, list):
            json_type = "array"
            json_keys = []
        elif json_parseable:
            json_type = type(parsed_body).__name__
            json_keys = []
        else:
            json_type = None
            json_keys = []
        evidence_response = EvidenceResponse(
            status_code=response.status_code,
            selected_headers={
                name: value
                for name, value in response.headers.items()
                if name.lower() in SELECTED_RESPONSE_HEADERS
            },
            body_sha256=_digest(sanitized_body),
            sanitized_body_excerpt=_excerpt(
                sanitized_body, config.limits.max_response_excerpt_bytes
            ),
            body_truncated=len(encoded_body) > config.limits.max_response_excerpt_bytes,
            json_parseable=json_parseable,
            json_type=json_type,
            json_top_level_keys=json_keys,
            duration_ms=duration_ms,
        )
    except (httpx.RequestError, ValueError) as exc:
        evidence_response = EvidenceResponse(
            duration_ms=(time.perf_counter() - started) * 1_000,
            transport_error=f"{type(exc).__name__}: {exc}",
        )

    return EvidenceRecord(
        actor=actor,
        iteration=iteration,
        scenario_or_rule_id=scenario_or_rule_id,
        request=evidence_request,
        response=evidence_response,
    )
