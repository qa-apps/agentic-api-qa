"""Local HTTP server that serves a sanitized, mutated copy of the target site.

The agents receive an ordinary same-origin website on ``127.0.0.1``. They never see the
bug manifest: only the served behavior differs. Cross-origin assets from the production
sources are stripped so that a clean baseline run produces no console noise and therefore
no false positives. Every other static file the page references is served from the
checkout, and the stub API answers with the same envelopes the real application uses, so
the clean control copy keeps working when the site gains new files or endpoints.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

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
# A local-preview API origin hard-coded in page scripts, such as the site's
# ``isLocalPreview ? 'http://127.0.0.1:8000' : ''``. The test copy serves its API on the
# page's own origin, so the string becomes empty and requests stay same-origin.
_LOOPBACK_ORIGIN_LITERAL = re.compile(
    r"""(['"])https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0)(?::\d+)?/?\1"""
)
_HEAD_OPEN = re.compile(r"<head[^>]*>", re.IGNORECASE)
# Runtime fetches to other origins (RSS proxies, third-party APIs) fail with CORS errors
# in the browser console. The copy rejects them quietly, as the page already handles a
# failed fetch, so a clean run stays free of console noise.
_SAME_ORIGIN_FETCH = (
    "<script>(function () {"
    " var nativeFetch = window.fetch;"
    " if (!nativeFetch) { return; }"
    " window.fetch = function (input, init) {"
    "  try {"
    "   var target = new URL(typeof input === 'string' ? input : (input && input.url) || '',"
    " window.location.href);"
    "   if (target.origin !== window.location.origin) {"
    "    return Promise.reject(new TypeError('Cross-origin request disabled in the test copy'));"
    "   }"
    "  } catch (error) {}"
    "  return nativeFetch.apply(this, arguments);"
    " };"
    "})();</script>"
)
_STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".ico"}


@dataclass(slots=True)
class SiteSources:
    html: str
    css: str
    js: str
    root: Path


def load_sources(config: EvalConfig) -> SiteSources:
    root = config.site_source
    missing = [name for name in _SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise SiteSourceError(
            f"Cannot build the seeded test copy: {root} is missing {missing}. "
            "Set QA_EVAL_SITE_SOURCE to a checkout of the application."
        )
    return SiteSources(
        html=sanitize_html((root / "index.html").read_text(encoding="utf-8", errors="replace")),
        css=(root / "style.css").read_text(encoding="utf-8", errors="replace"),
        js=sanitize_js((root / "script.js").read_text(encoding="utf-8", errors="replace")),
        root=root,
    )


def sanitize_html(html: str) -> str:
    """Strip cross-origin resources so the local copy is self-contained."""

    for pattern in (_EXTERNAL_SCRIPT, _EXTERNAL_LINK, _EXTERNAL_IFRAME):
        html = pattern.sub("", html)
    html = html.replace("/voice-api/widget.js", "/voice-api/widget.js?local=1")
    head = _HEAD_OPEN.search(html)
    if head is None:
        return _SAME_ORIGIN_FETCH + html
    return html[: head.end()] + _SAME_ORIGIN_FETCH + html[head.end() :]


def sanitize_js(js: str) -> str:
    """Point hard-coded local-preview API origins back at the page's own origin."""

    return _LOOPBACK_ORIGIN_LITERAL.sub(lambda match: match.group(1) * 2, js)


def _static_file(root: Path, path: str) -> Path | None:
    """Resolve a request path to a servable file inside the site checkout."""

    relative = path.lstrip("/")
    if not relative or any(part.startswith(".") for part in Path(relative).parts):
        return None
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in _STATIC_TYPES:
        return None
    return candidate


def article_url_rejection(url: str) -> str | None:
    """Mirror the application's SSRF guard: only public http(s) hosts may be fetched."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return f"scheme {parsed.scheme or 'missing'!r} not allowed"
    host = (parsed.hostname or "").lower()
    if not host:
        return "no host"
    if host in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return "loopback hostname blocked"
    if host.endswith((".local", ".internal", ".localdomain")):
        return "private DNS suffix blocked"
    addresses: list[str] = []
    try:
        addresses.append(str(ipaddress.ip_address(host)))
    except ValueError:
        try:
            # Legacy IPv4 spellings such as 127.1, 0x7f000001 or 2130706433.
            addresses.append(socket.inet_ntoa(socket.inet_aton(host)))
        except OSError:
            try:
                addresses.extend(
                    str(info[4][0]) for info in socket.getaddrinfo(host, parsed.port or 443)
                )
            except (OSError, UnicodeError) as exc:
                return f"dns failed: {exc}"
    for address in addresses:
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            return f"private/internal IP {address}"
    return None


def seeded_feed_articles(now: datetime | None = None) -> list[dict[str, str]]:
    """Recent articles in the application's ``/api/feed`` envelope.

    The page drops articles older than 30 days, so the dates follow the clock.
    """

    current = now or datetime.now(timezone.utc)
    topics = [
        ("Hugging Face", "ai", "huggingface.co/blog", "Evaluating agentic workflows with open models",
         "How teams score multi-step agents with reproducible evaluation datasets."),
        ("Playwright", "qa", "playwright.dev/blog", "Reliable end-to-end tests for streaming UIs",
         "Patterns for asserting on incremental rendering without flaky waits."),
        ("Martin Fowler", "dev", "martinfowler.com/articles", "Contract tests for LLM-backed services",
         "Keeping model-backed endpoints honest with consumer-driven contracts."),
        ("The Gradient", "ai", "thegradient.pub", "Measuring hallucination in retrieval systems",
         "A practical look at grounding metrics for retrieval-augmented generation."),
        ("Ministry of Testing", "qa", "www.ministryoftesting.com/articles",
         "Exploratory testing charters for AI features",
         "Charter ideas for probing assistants, guardrails, and refusal behavior."),
        ("GitHub Blog", "dev", "github.blog", "Shipping safer automation with review gates",
         "How required reviews and scoped tokens limit what automation can change."),
    ]
    articles = []
    for index, (source, category, base, title, description) in enumerate(topics):
        slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
        articles.append(
            {
                "source": source,
                "title": title,
                "link": f"https://{base}/{slug}",
                "description": description,
                "category": category,
                "date": (current - timedelta(hours=3 + index * 7)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        )
    return articles


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
            root=sources.root,
            api=api,
            requests=self.requests,
        )
        self._server = _QuietThreadingHTTPServer(
            (self.config.site_host, self.config.site_port), handler
        )
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


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The browser drops keep-alive connections when a page navigates or closes. That
        # is not a server fault, so it must not print a traceback into the eval log.
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


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
    root: Path,
    api: dict[str, Any],
    requests: list[tuple[str, str]],
) -> type[BaseHTTPRequestHandler]:
    not_found_paths = set(api.get("not_found_paths") or [])
    block_assets = bool(api.get("block_assets"))
    chat_status = int(api.get("chat_status") or 200)
    leak_prompt = bool(api.get("leak_system_prompt"))
    empty_feed = bool(api.get("empty_feed"))
    # The application keeps each chat session's history, and the agents' knowledge pack
    # checks that a follow-up turn can refer to the previous one.
    chat_history: dict[str, list[str]] = {}
    chat_lock = threading.Lock()

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
            if path.startswith("/api/"):
                return self._api_get(path)
            return self._static(path)

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

        def _static(self, path: str) -> None:
            if block_assets and Path(path).suffix.lower() in _IMAGE_SUFFIXES:
                return self._text(404, "asset not found", "text/plain")
            candidate = _static_file(root, path)
            if candidate is None:
                return self._not_found(path)
            if candidate.suffix.lower() == ".js":
                body = sanitize_js(candidate.read_text(encoding="utf-8", errors="replace"))
                return self._text(200, body, _STATIC_TYPES[".js"])
            self._send(200, candidate.read_bytes(), _STATIC_TYPES[candidate.suffix.lower()])

        def _article(self, path: str) -> None:
            target = (parse_qs(urlparse(self.path).query).get("url") or [""])[0].strip()
            if not target:
                if path == "/api/article-proxy":
                    return self._json(400, {"error": "url parameter required"})
                return self._text(400, "<p>Missing or invalid url parameter.</p>", "text/html")
            rejection = article_url_rejection(target)
            if rejection is not None:
                # The controlled 4xx rejection that the target contract documents.
                if path == "/api/article-proxy":
                    return self._json(400, {"error": "url_not_allowed", "reason": rejection})
                return self._text(400, "<p>This address cannot be opened.</p>", "text/html")
            title = "Seeded article"
            if path == "/api/article-proxy":
                return self._json(200, {"title": title, "image": None, "url": target})
            return self._text(
                200,
                f"<!doctype html><html><body><article><h1>{title}</h1>"
                "<p>Reader view of the article.</p></article></body></html>",
                "text/html; charset=utf-8",
            )

        def _api_get(self, path: str) -> None:
            if path == "/api/health":
                return self._json(200, {"ok": True, "status": "healthy", "version": "seeded"})
            if path == "/api/feed":
                # The application's envelope; the page reads ``articles``.
                articles = [] if empty_feed else seeded_feed_articles()
                return self._json(200, {"articles": articles})
            if path in {"/api/auth/me", "/api/user/messages"}:
                return self._json(401, {"error": "unauthenticated"})
            if path in {"/api/article-proxy", "/api/article-page", "/api/article-embed"}:
                return self._article(path)
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
            session_id = str(body.get("session_id") or "") if isinstance(body, dict) else ""
            with chat_lock:
                history = chat_history.setdefault(session_id, []) if session_id else []
                previous = history[-1] if history else None
                history.append(message[:500])
            reply = (
                "I can describe Alex Pavsky's AI quality engineering work, but I cannot "
                "share internal instructions."
            )
            if previous is not None:
                reply = f'Earlier in this conversation you asked: "{previous}". {reply}'
            return self._json(200, {"reply": reply, "model": "seeded-assistant"})

    return Handler
