"""Official Playwright MCP adapter and bounded UI smoke executor."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from langsmith import traceable
from mcp import Client, StdioServerParameters

from agentic_api_qa.models import RunConfig, Severity, UICaseResult, UISmokeCase


PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@0.0.81"
BASE_TOOLS = {
    "browser_navigate",
    "browser_resize",
    "browser_snapshot",
    "browser_click",
    "browser_fill_form",
    "browser_press_key",
    "browser_console_messages",
    "browser_network_requests",
    "browser_take_screenshot",
}
VIDEO_TOOLS = {"browser_start_video", "browser_stop_video"}
ALLOWED_TOOLS = BASE_TOOLS | VIDEO_TOOLS


def _text(result) -> str:
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


def _assert_same_origin(base_url: str, target_url: str) -> None:
    base = urlparse(base_url)
    target = urlparse(target_url)
    if (base.scheme, base.netloc) != (target.scheme, target.netloc):
        raise ValueError(f"MCP navigation blocked outside target origin: {target_url}")


@traceable(run_type="tool", name="playwright_mcp_call")
async def _call(client: Client, tool: str, arguments: dict) -> tuple[str, bool]:
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Playwright MCP tool is not allowlisted: {tool}")
    result = await client.call_tool(tool, arguments)
    return _text(result), bool(result.is_error)


def _server_parameters(run_config: RunConfig, artifact_dir: Path) -> StdioServerParameters:
    video_enabled = run_config.ui_video != "off"
    return StdioServerParameters(
        command="npx",
        args=[
            "--yes",
            PLAYWRIGHT_MCP_PACKAGE,
            *(["--headless"] if run_config.ui_headless else []),
            *(["--caps=devtools"] if video_enabled else []),
            "--isolated",
            "--browser",
            "chrome",
            "--output-dir",
            str(artifact_dir),
        ],
        cwd=artifact_dir,
    )


@traceable(run_type="tool", name="playwright_mcp_smoke_suite")
async def execute_ui_smoke_suite(
    *, run_config: RunConfig, run_id: str, cases: list[UISmokeCase]
) -> tuple[list[UICaseResult], int]:
    """Run independent, read-only UI cases through the official MCP server."""

    artifact_dir = (Path(run_config.ui_artifact_dir) / run_id).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    params = _server_parameters(run_config, artifact_dir)
    results: list[UICaseResult] = []
    tool_calls = 0
    async with Client(params) as client:
        await _verify_tools(client, video_enabled=run_config.ui_video != "off")
        for case in cases[: run_config.limits.hard_case_limit]:
            result, calls = await _execute_case(
                client=client,
                run_config=run_config,
                artifact_dir=artifact_dir,
                case=case,
                calls_so_far=tool_calls,
            )
            results.append(result)
            tool_calls += calls
    return results, tool_calls


async def _verify_tools(client: Client, *, video_enabled: bool) -> None:
    available = {tool.name for tool in (await client.list_tools()).tools}
    required = BASE_TOOLS | (VIDEO_TOOLS if video_enabled else set())
    missing = required - available
    if missing:
        raise RuntimeError(f"Playwright MCP missing required tools: {sorted(missing)}")


def _severity(case: UISmokeCase, failures: list[str], console_errors: list[str]) -> Severity:
    if not failures:
        return Severity.INFO
    if console_errors or case.category in {"smoke", "authentication", "chat"}:
        return Severity.HIGH
    return Severity.MEDIUM


async def _execute_case(
    *,
    client: Client,
    run_config: RunConfig,
    artifact_dir: Path,
    case: UISmokeCase,
    calls_so_far: int,
) -> tuple[UICaseResult, int]:
    """Execute one case and always attempt evidence capture, including on failure."""

    started = time.perf_counter()
    used: list[str] = []
    failures: list[str] = []
    snapshot = ""
    console_text = ""
    calls = 0
    screenshot_name = f"{case.case_id}.png"
    video_name = f"{case.case_id}.webm"
    video_path = artifact_dir / video_name
    video_started = False

    async def invoke(tool: str, arguments: dict, *, evidence: bool = False) -> str:
        nonlocal calls
        if not evidence and calls_so_far + calls >= run_config.limits.ui_tool_calls:
            raise RuntimeError("UI MCP tool-call budget exhausted")
        text, is_error = await _call(client, tool, arguments)
        calls += 1
        used.append(tool)
        if is_error:
            raise RuntimeError(f"{tool}: {text[:300]}")
        return text

    if run_config.ui_video != "off":
        try:
            await invoke(
                "browser_start_video",
                {
                    "filename": video_name,
                    "size": {"width": case.viewport_width, "height": case.viewport_height},
                },
                evidence=True,
            )
            video_started = True
        except Exception as exc:
            failures.append(f"VideoCaptureError: {exc}")

    try:
        url = urljoin(f"{str(run_config.base_url).rstrip('/')}/", case.path.lstrip("/"))
        _assert_same_origin(str(run_config.base_url), url)
        await invoke(
            "browser_resize", {"width": case.viewport_width, "height": case.viewport_height}
        )
        await invoke("browser_navigate", {"url": url})
        for action in case.actions:
            await invoke(action["tool"], action.get("arguments", {}))
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    # Evidence calls are mandatory once a case starts. They remain available even
    # when an action failed or the ordinary action budget was just exhausted.
    try:
        snapshot = await invoke("browser_snapshot", {"depth": 12}, evidence=True)
    except Exception as exc:
        failures.append(f"SnapshotCaptureError: {exc}")
    try:
        console_text = await invoke(
            "browser_console_messages", {"level": "error", "all": False}, evidence=True
        )
    except Exception as exc:
        failures.append(f"ConsoleCaptureError: {exc}")
    try:
        await invoke(
            "browser_take_screenshot",
            {"filename": screenshot_name, "fullPage": False, "scale": "css"},
            evidence=True,
        )
    except Exception as exc:
        failures.append(f"ScreenshotCaptureError: {exc}")

    folded = snapshot.casefold()
    missing_text = [value for value in case.expected_text if value.casefold() not in folded]
    if missing_text:
        failures.append(f"Expected visible text not found: {missing_text}")
    console_errors: list[str] = []
    if "errors: 0" not in console_text.casefold() and console_text.strip():
        console_errors = [line for line in console_text.splitlines() if line.strip()][-10:]
        failures.append("Browser console reported errors")
    if video_started:
        try:
            await invoke("browser_stop_video", {}, evidence=True)
        except Exception as exc:
            failures.append(f"VideoCaptureError: {exc}")

    retain_video = run_config.ui_video == "on" or bool(failures)
    if video_path.exists() and not retain_video:
        video_path.unlink()
    screenshot_path = artifact_dir / screenshot_name
    video_captured = retain_video and video_path.exists()
    severity = _severity(case, failures, console_errors)
    return (
        UICaseResult(
            case_id=case.case_id,
            name=case.name,
            category=case.category,
            passed=not failures,
            reason="Passed MCP UI checks." if not failures else "; ".join(failures),
            screenshot_path=str(screenshot_path) if screenshot_path.exists() else None,
            screenshot_captured=screenshot_path.exists(),
            video_path=str(video_path) if video_captured else None,
            video_captured=video_captured,
            severity=severity,
            investigation_required=severity in {Severity.HIGH, Severity.CRITICAL},
            snapshot_excerpt=snapshot[:4_000],
            console_errors=console_errors,
            mcp_tools=used,
            duration_ms=(time.perf_counter() - started) * 1_000,
        ),
        calls,
    )


@traceable(run_type="tool", name="playwright_mcp_test_case")
async def execute_ui_smoke_case(
    *, run_config: RunConfig, run_id: str, case: UISmokeCase, calls_so_far: int = 0
) -> tuple[UICaseResult, int]:
    """Execute one independently isolated MCP test case for the agentic loop."""

    artifact_dir = (Path(run_config.ui_artifact_dir) / run_id).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    params = _server_parameters(run_config, artifact_dir)
    async with Client(params) as client:
        await _verify_tools(client, video_enabled=run_config.ui_video != "off")
        return await _execute_case(
            client=client,
            run_config=run_config,
            artifact_dir=artifact_dir,
            case=case,
            calls_so_far=calls_so_far,
        )
