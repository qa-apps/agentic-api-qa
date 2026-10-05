import json
from pathlib import Path

from agentic_api_qa.models import (
    Actor,
    CandidateFinding,
    FinalReport,
    FindingKind,
    JudgeDecision,
    JudgeVerdict,
    OverallResult,
    Severity,
)
from agentic_api_qa.slack_summary import build_payload, main

SAMPLE = Path(__file__).resolve().parents[1] / "reports" / "sample-report.json"


def _sample() -> FinalReport:
    return FinalReport.model_validate(json.loads(SAMPLE.read_text(encoding="utf-8")))


def _blocks(payload: dict) -> list[dict]:
    return payload["attachments"][0]["blocks"]


def test_summary_names_the_agents_and_links_the_full_report_first() -> None:
    payload = build_payload(
        _sample(),
        channel="C123",
        report_url="https://qa-apps.github.io/agentic-api-qa/runs/7/",
        run_url="https://github.com/qa-apps/agentic-api-qa/actions/runs/1",
    )
    blocks = _blocks(payload)

    assert "AP Agents · Agentic QA" in blocks[0]["text"]["text"]
    assert "Explorer, Adversary and UI agents" in blocks[1]["elements"][0]["text"]
    buttons = blocks[-1]["elements"]
    assert [button["text"]["text"] for button in buttons] == ["Open full report", "GitHub run"]
    assert buttons[0]["url"].endswith("/runs/7/")


def test_summary_lists_only_judge_confirmed_findings() -> None:
    report = _sample()
    kept = CandidateFinding(
        source=Actor.EXPLORER,
        kind=FindingKind.API,
        severity=Severity.HIGH,
        title="Chat API returns HTTP 500",
        expected="200",
        actual="500",
    )
    dropped = CandidateFinding(
        source=Actor.EXPLORER,
        kind=FindingKind.API,
        severity=Severity.LOW,
        title="Rejected by the judge",
        expected="a",
        actual="b",
    )
    decisions = [
        JudgeDecision(
            finding_id=finding.finding_id,
            verdict=verdict,
            confidence=0.9,
            skeptical_challenge="x",
            reasoning_summary="x",
            investigation_reason="x",
        )
        for finding, verdict in ((kept, JudgeVerdict.CONFIRMED), (dropped, JudgeVerdict.REJECTED))
    ]
    report = report.model_copy(
        update={
            "candidate_findings": [kept, dropped],
            "judge_decisions": decisions,
            "run": report.run.model_copy(update={"overall_result": OverallResult.FAIL}),
        }
    )

    text = json.dumps(build_payload(report, channel="C123"))

    assert "[HIGH] Chat API returns HTTP 500" in text
    assert "Rejected by the judge" not in text
    assert "1 of 2 candidates" in text
    assert "FAIL" in _blocks(build_payload(report, channel="C123"))[0]["text"]["text"]


def test_missing_token_skips_without_failing_the_run(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    assert main(["--report", str(SAMPLE), "--channel", "C123"]) == 0
