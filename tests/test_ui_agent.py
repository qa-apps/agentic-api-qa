from types import SimpleNamespace

import pytest

from agentic_api_qa.llm import UIAgentChoice, UsageDelta
from agentic_api_qa.mcp_browser import _execute_case, _server_parameters
from agentic_api_qa.models import (
    QAState,
    RunConfig,
    SafetyLimits,
    Severity,
    UICaseResult,
    UISmokeCase,
)
from agentic_api_qa.nodes import ui_explorer_agent
from agentic_api_qa.ui_catalog import alexpavsky_ui_smoke_catalog


def result_for(case_id: str, *, severity: Severity = Severity.INFO) -> UICaseResult:
    return UICaseResult(
        case_id=case_id,
        name=case_id,
        category="smoke",
        passed=severity == Severity.INFO,
        reason="evidence",
        severity=severity,
        investigation_required=severity in {Severity.HIGH, Severity.CRITICAL},
    )


@pytest.mark.asyncio
async def test_ui_agent_stops_at_soft_limit_without_serious_bug() -> None:
    catalog = alexpavsky_ui_smoke_catalog()
    state = QAState(
        config=RunConfig(limits=SafetyLimits(soft_case_limit=20, hard_case_limit=30)),
        ui_results=[result_for(case.case_id) for case in catalog[:20]],
    )

    update = await ui_explorer_agent(state)

    assert update["ui_done"] is True
    assert update["ui_stop_reason"] == "soft_limit_reached_without_serious_bug"
    assert len(update["ui_skipped_case_ids"]) == 10


@pytest.mark.asyncio
async def test_ui_agent_can_investigate_past_soft_limit_with_cited_high_bug(
    monkeypatch,
) -> None:
    catalog = alexpavsky_ui_smoke_catalog()
    serious_id = catalog[0].case_id
    results = [result_for(case.case_id) for case in catalog[:20]]
    results[0] = result_for(serious_id, severity=Severity.HIGH)
    state = QAState(
        config=RunConfig(limits=SafetyLimits(soft_case_limit=20, hard_case_limit=30)),
        ui_results=results,
        ui_serious_bug_case_ids=[serious_id],
    )

    async def choose(*_args, **_kwargs):
        return UIAgentChoice(
            action="execute",
            case_id=catalog[20].case_id,
            summary="Investigate the serious homepage failure at another viewport.",
            investigation_reason=f"Follow-up for HIGH failure {serious_id}",
            usage=UsageDelta(total_tokens=10),
        )

    monkeypatch.setattr("agentic_api_qa.nodes.choose_next_ui_case", choose)
    update = await ui_explorer_agent(state)

    assert update["ui_pending_case"].case_id == catalog[20].case_id
    assert update.get("ui_done") is not True
    assert serious_id in update["ui_investigation_reasons"][-1]


@pytest.mark.asyncio
async def test_ui_agent_never_exceeds_hard_limit() -> None:
    catalog = alexpavsky_ui_smoke_catalog()
    state = QAState(
        config=RunConfig(limits=SafetyLimits(soft_case_limit=20, hard_case_limit=30)),
        ui_results=[result_for(case.case_id, severity=Severity.HIGH) for case in catalog],
        ui_serious_bug_case_ids=[catalog[0].case_id],
    )

    update = await ui_explorer_agent(state)

    assert update["ui_done"] is True
    assert update["ui_stop_reason"] == "hard_limit_reached"


class FakeMCPClient:
    def __init__(self, artifact_dir) -> None:
        self.artifact_dir = artifact_dir
        self.calls = []
        self.video_name = None

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        if tool == "browser_start_video":
            self.video_name = arguments["filename"]
        elif tool == "browser_stop_video":
            (self.artifact_dir / self.video_name).write_bytes(b"webm")
        elif tool == "browser_take_screenshot":
            (self.artifact_dir / arguments["filename"]).write_bytes(b"png")
        text = {
            "browser_snapshot": "Home",
            "browser_console_messages": "Errors: 0",
        }.get(tool, "ok")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            is_error=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_text", "passed", "video_captured"),
    [(["Home"], True, False), (["Missing"], False, True)],
)
async def test_ui_video_is_retained_only_for_failed_cases(
    tmp_path, expected_text, passed, video_captured
) -> None:
    client = FakeMCPClient(tmp_path)
    result, _ = await _execute_case(
        client=client,
        run_config=RunConfig(ui_video="retain-on-failure"),
        artifact_dir=tmp_path,
        case=UISmokeCase(
            case_id="ui-video",
            name="Video evidence",
            category="smoke",
            objective="Capture failure evidence",
            expected_text=expected_text,
        ),
        calls_so_far=0,
    )

    assert result.passed is passed
    assert result.video_captured is video_captured
    assert bool(result.video_path) is video_captured
    assert (tmp_path / "ui-video.webm").exists() is video_captured
    assert [tool for tool, _ in client.calls].count("browser_start_video") == 1
    assert [tool for tool, _ in client.calls].count("browser_stop_video") == 1


def test_video_mode_enables_playwright_devtools_capability(tmp_path) -> None:
    params = _server_parameters(RunConfig(ui_video="retain-on-failure"), tmp_path)

    assert "--caps=devtools" in params.args
