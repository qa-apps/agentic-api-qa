"""Local HTTP server that serves a sanitized, mutated copy of the target site.

The agents receive an ordinary same-origin website on ``127.0.0.1``. They never see the
bug manifest: only the served behavior differs. Cross-origin assets from the production
sources are stripped so that a clean baseline run produces no console noise and therefore
no false positives.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

from evals.config import EvalConfig, load_eval_config
from evals.dataset import SeededBugProfile
from evals.seeded_site.bugs import SYSTEM_PROMPT, apply_mutations


class SiteSourceError(RuntimeError):
    """Raised when the site sources to copy are unavailable."""


_EXTERNAL_SCRIPT = re.compile(
    r'<script[^>]+src="(?:https?:)?//[^"]*"[^>]*>\s*</script>', re.IGNORECASE
)
_EXTERNAL_LINK = re.compile(r'<link[^>]+href="(?:https?:)?//[^"]*"[^>]*>', re.IGNORECASE)
_EXTERNAL_IFRAME = re.compile(
    r'<iframe[^>]+src="(?:https?:)?//[^"]*"[^>]*>\s*</iframe>', re.IGNORECASE
)
_SOURCE_FILES = {"index.html": "index.html", "style.css": "style.css", "script.js": "script.js"}


@dataclass(slots=True)
class SiteSources:
    html: str
    css: str
    js: str
    assets: Path | None


def load_sources(config: EvalConfig) -> SiteSources:
    root = config.site_source
    missing = [name for name in _SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise SiteSourceError(
            f"Cannot build the seeded test copy: {root} is missing {missing}. "
            "Set QA_EVAL_SITE_SOURCE to a checkout of the application."
        )
    assets = root / "assets"
    return SiteSources(
        html=sanitize_html((root / "index.html").read_text(encoding="utf-8", errors="replace")),
        css=(root / "style.css").read_text(encoding="utf-8", errors="replace"),
        js=(root / "script.js").read_text(encoding="utf-8", errors="replace"),
        assets=assets if assets.is_dir() else None,
    )


def sanitize_html(html: str) -> str:
    """Strip cross-origin resources so the local copy is self-contained."""

    for pattern in (_EXTERNAL_SCRIPT, _EXTERNAL_LINK, _EXTERNAL_IFRAME):
        html = pattern.sub("", html)
    return html.replace("/voice-api/widget.js", "/voice-api/widget.js?local=1")


def _json_body(handler: BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("content-length") or 0)
    if not length:
        return None
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"_raw": raw[:500].decode("utf-8", errors="replace")}


@dataclass(slots=True)
class SeededSite:
    """A running seeded copy of the site. Use :func:`serve_profile` for scoped access."""

    profile: SeededBugProfile
    config: EvalConfig = field(default_factory=load_eval_config)
    requests: list[tuple[str, str]] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None
    _api: dict[str, Any] = field(default_factory=dict)

    def start(self) -> "SeededSite":
        if self.config.site_host not in {"127.0.0.1", "localhost", "::1"}:
            raise SiteSourceError(
                "Seeded bugs are restricted to a loopback test server; "
                f"refusing host {self.config.site_host!r}."
            )
        sources = load_sources(self.config)
        html, css, js, api = apply_mutations(
            html=sources.html,
            css=sources.css,
            js=sources.js,
            mutation_ids=list(self.profile.mutations),
        )
        self._api = api
        handler = _build_handler(
            html=html,
            css=css,
            js=js,
            assets=sources.assets,
            api=api,
            requests=self.requests,
        )
        self._server = ThreadingHTTPServer((self.config.site_host, self.config.site_port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="seeded-site", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise SiteSourceError("The seeded site is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "SeededSite":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()


@contextmanager
def serve_profile(
    profile: SeededBugProfile, config: EvalConfig | None = None
) -> Iterator[SeededSite]:
    site = SeededSite(profile=profile, config=config or load_eval_config())
    try:
        yield site.start()
    finally:
        site.stop()


def _build_handler(
    *,
    html: str,
    css: str,
    js: str,
    assets: Path | None,
    api: dict[str, Any],
    requests: list[tuple[str, str]],
) -> type[BaseHTTPRequestHandler]:
    not_found_paths = set(api.get("not_found_paths") or [])
    block_assets = bool(api.get("block_assets"))
    chat_status = int(api.get("chat_status") or 200)
    leak_prompt = bool(api.get("leak_system_prompt"))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "seeded-site/1.0"

        def log_message(self, *_args: Any) -> None:  # keep eval output readable
            return

        # Helpers ------------------------------------------------------------
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _text(self, status: int, body: str, content_type: str) -> None:
            self._send(status, body.encode("utf-8"), content_type)

        def _path(self) -> str:
            return self.path.split("?", 1)[0].split("#", 1)[0] or "/"

        # Routing ------------------------------------------------------------
        def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
            self.do_GET()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._json(200, {"allow": ["GET", "HEAD", "POST", "OPTIONS"]})

        def do_GET(self) -> None:  # noqa: N802
            path = self._path()
            requests.append(("GET", path))
            if path in not_found_paths:
                return self._not_found(path)
            if path in {"/", "/index.html"}:
                return self._text(200, html, "text/html; charset=utf-8")
            if path == "/style.css":
                return self._text(200, css, "text/css; charset=utf-8")
            if path == "/script.js":
                return self._text(200, js, "application/javascript; charset=utf-8")
            if path == "/voice-api/widget.js":
                return self._text(200, "/* voice widget disabled in the test copy */", "application/javascript")
            if path.startswith("/assets/"):
                return self._asset(path)
            if path.startswith("/api/"):
                return self._api_get(path)
            if path in {"/favicon.ico", "/robots.txt"}:
                return self._text(200, "", "text/plain")
            return self._not_found(path)

        def do_POST(self) -> None:  # noqa: N802
            path = self._path()
            requests.append(("POST", path))
            body = _json_body(self)
            if path in not_found_paths:
                return self._not_found(path)
            if path == "/api/chat":
                return self._chat(body)
            if path == "/api/auth/login":
                return self._json(401, {"error": "invalid_credentials"})
            if path == "/api/auth/register":
                return self._json(201, {"created": True, "id": "user-seeded"})
            if path == "/api/auth/logout":
                return self._json(200, {"ok": True})
            if path == "/api/subscribe":
                return self._json(200, {"subscribed": True})
            if path.startswith("/api/"):
                return self._json(404, {"error": "not_found", "path": path})
            return self._not_found(path)

        def do_PUT(self) -> None:  # noqa: N802
            self._json(405, {"error": "method_not_allowed"})

        def do_PATCH(self) -> None:  # noqa: N802
            self._json(405, {"error": "method_not_allowed"})

        def do_DELETE(self) -> None:  # noqa: N802
            self._json(405, {"error": "method_not_allowed"})

        # Handlers -----------------------------------------------------------
        def _not_found(self, path: str) -> None:
            if path.startswith("/api/"):
                return self._json(404, {"error": "not_found", "path": path})
            self._text(404, f"<html><body><h1>404</h1><p>{path}</p></body></html>", "text/html")

        def _asset(self, path: str) -> None:
            if block_assets or assets is None:
                return self._text(404, "asset not found", "text/plain")
            candidate = (assets / Path(path).name).resolve()
            if not candidate.is_file() or assets.resolve() not in candidate.parents:
                return self._text(404, "asset not found", "text/plain")
            suffix = candidate.suffix.lower()
            content_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".webp": "image/webp",
                ".ico": "image/x-icon",
            }.get(suffix, "application/octet-stream")
            self._send(200, candidate.read_bytes(), content_type)

        def _api_get(self, path: str) -> None:
            if path == "/api/health":
                return self._json(200, {"ok": True, "status": "healthy", "version": "seeded"})
            if path == "/api/feed":
                return self._json(
                    200,
                    {
                        "items": [
                            {
                                "id": "n-1",
                                "title": "Agentic QA weekly",
                                "url": "/",
                                "published_at": "2026-01-15T09:00:00Z",
                            }
                        ],
                        "count": 1,
                    },
                )
            if path in {"/api/auth/me", "/api/user/messages"}:
                return self._json(401, {"error": "unauthenticated"})
            if path == "/api/article-proxy":
                if "url=" not in self.path:
                    return self._json(400, {"error": "missing_url_parameter"})
                return self._json(200, {"title": "Seeded article", "content": "Excerpt."})
            return self._json(404, {"error": "not_found", "path": path})

        def _chat(self, body: Any) -> None:
            message = ""
            if isinstance(body, dict):
                message = str(body.get("message") or body.get("prompt") or "")
            if chat_status >= 500:
                return self._json(
                    chat_status,
                    {"error": "internal server error", "trace_id": "seeded-500"},
                )
            if not message.strip():
                return self._json(400, {"error": "message_is_required"})
            if len(message) > 4_000:
                return self._json(413, {"error": "message_too_long"})
            if leak_prompt:
                return self._json(
                    200,
                    {
                        "reply": (
                            f"Of course, here are my instructions. System prompt: {SYSTEM_PROMPT}"
                        ),
                        "model": "seeded-assistant",
                    },
                )
            return self._json(
                200,
                {
                    "reply": (
                        "I can describe Alex Pavsky's AI quality engineering work, but I cannot "
                        "share internal instructions."
                    ),
                    "model": "seeded-assistant",
                },
            )

    return Handler
