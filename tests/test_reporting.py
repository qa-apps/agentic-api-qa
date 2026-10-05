from datetime import datetime, timezone

from agentic_api_qa.models import (
    Environment,
    FinalReport,
    HappyPathReport,
    HappyPathResult,
    OverallResult,
    ReportMetrics,
    ReportRun,
    UICaseResult,
)
from agentic_api_qa.reporting import (
    publish_report_bundle,
    render_html_report,
    render_ui_execution_report,
)


def test_html_report_embeds_screenshot_for_passed_case(tmp_path) -> None:
    screenshot = tmp_path / "passed.png"
    screenshot.write_bytes(b"png")
    now = datetime.now(timezone.utc)
    metrics = ReportMetrics(
        happy_path_passed=0,
        happy_path_failed=0,
        guardrails_held=0,
        guardrails_breached=0,
        guardrails_inconclusive=0,
        pipeline_errors=0,
        explorer_iterations=0,
        adversary_iterations=0,
        explorer_tool_calls=0,
        adversary_tool_calls=0,
        total_tool_calls=0,
        ui_cases_passed=1,
        ui_screenshots=1,
    )
    report = FinalReport(
        run=ReportRun(
            run_id="run-report",
            target="https://www.alexpavsky.com",
            environment=Environment.PRODUCTION,
            started_at=now,
            finished_at=now,
            duration_ms=1,
            overall_result=OverallResult.PASS,
            production_read_only=True,
        ),
        happy_path=HappyPathReport(status=HappyPathResult.PASS, steps=[]),
        adversarial=[],
        summary="Passed.",
        metrics=metrics,
        evidence=[],
        audit_log=[],
        pipeline_errors=[],
        ui_smoke=[
            UICaseResult(
                case_id="ui-pass",
                name="Passed UI case",
                category="smoke",
                passed=True,
                reason="Passed",
                screenshot_path=str(screenshot),
                screenshot_captured=True,
            )
        ],
    )

    destination = render_html_report(report, tmp_path / "report.html")
    rendered = destination.read_text()

    assert "ui-pass" in rendered
    assert "<img" in rendered
    assert "passed.png" in rendered
    assert "data:image/png;base64," in rendered
    assert 'height:360px' in rendered
    assert 'target="_blank"' in rendered


def test_standalone_ui_report_contains_images_and_failure_video(tmp_path) -> None:
    screenshot = tmp_path / "evidence.png"
    screenshot.write_bytes(b"png")
    video = tmp_path / "failure.webm"
    video.write_bytes(b"webm")
    payload = {
        "run_id": "run-ui",
        "plan_summary": "Run both cases.",
        "results": [
            {
                "case_id": f"ui-{index}",
                "name": f"Case {index}",
                "category": "smoke",
                "passed": index == 1,
                "reason": "Passed MCP UI checks." if index == 1 else "Expected text missing.",
                "screenshot_path": str(screenshot),
                "screenshot_captured": True,
                "video_path": str(video) if index == 2 else None,
                "video_captured": index == 2,
                "mcp_tools": ["browser_take_screenshot"],
            }
            for index in (1, 2)
        ],
        "pipeline_errors": [],
    }

    destination = render_ui_execution_report(payload, tmp_path / "ui-report.html")
    rendered = destination.read_text()

    assert "1/2 passed" in rendered
    assert "ui-1" in rendered
    assert "ui-2" in rendered
    assert rendered.count("data:image/png;base64,") == 4
    assert '<video controls preload="metadata">' in rendered
    assert "videos/run-ui/failure.webm" in rendered
    assert (tmp_path / "videos" / "run-ui" / "failure.webm").read_bytes() == b"webm"


def test_publish_report_bundle_updates_stable_latest_entry_points(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    report = FinalReport(
        run=ReportRun(
            run_id="run-latest",
            target="https://www.alexpavsky.com",
            environment=Environment.PRODUCTION,
            started_at=now,
            finished_at=now,
            duration_ms=1,
            overall_result=OverallResult.PASS,
            production_read_only=True,
        ),
        happy_path=HappyPathReport(status=HappyPathResult.PASS, steps=[]),
        adversarial=[],
        summary="Passed.",
        metrics=ReportMetrics(
            happy_path_passed=0,
            happy_path_failed=0,
            guardrails_held=0,
            guardrails_breached=0,
            guardrails_inconclusive=0,
            pipeline_errors=0,
            explorer_iterations=0,
            adversary_iterations=0,
            explorer_tool_calls=0,
            adversary_tool_calls=0,
            total_tool_calls=0,
        ),
        evidence=[],
        audit_log=[],
        pipeline_errors=[],
    )

    paths = publish_report_bundle(report, tmp_path / "reports")

    assert paths["latest_html"].name == "index.html"
    assert paths["latest_html"].is_file()
    assert paths["latest_json"].is_file()
    assert "run-latest" in paths["latest_html"].read_text()
