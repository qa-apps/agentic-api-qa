"""Local Playwright-style HTTP server for the latest visual QA report."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

from agentic_api_qa.reporting import report_directory


def report_server_url() -> str:
    host = os.getenv("QA_REPORT_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("QA_REPORT_SERVER_PORT", "9324"))
    return f"http://{host}:{port}/"


def _is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.3) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_report_server(directory: Path | None = None) -> str | None:
    """Start a localhost-only report server unless it is already available."""

    enabled = os.getenv("QA_REPORT_SERVER_ENABLED", "true").lower() == "true"
    if not enabled or os.getenv("CI"):
        return None
    reports = (directory or report_directory()).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    latest = reports / "index.html"
    if not latest.is_file():
        return None
    url = report_server_url()
    if _is_ready(url):
        return url

    host = os.getenv("QA_REPORT_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("QA_REPORT_SERVER_PORT", "9324"))
    with (reports / ".report-server.log").open("ab") as log:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                host,
                "--directory",
                str(reports),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    for _ in range(30):
        if _is_ready(url):
            return url
        time.sleep(0.1)
    raise RuntimeError(f"Visual report server did not start at {url}")


def main() -> None:
    url = ensure_report_server()
    if not url:
        raise SystemExit(f"No visual report found in {report_directory()}")
    webbrowser.open(url)
    print(f"Opened visual report: {url}")

