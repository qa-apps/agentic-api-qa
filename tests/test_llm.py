import json

import pytest

from agentic_api_qa.llm import (
    choose_next_action,
    evaluate_candidate_findings,
    reflect_on_evidence,
)
from agentic_api_qa.models import (
    Actor,
    CandidateFinding,
    Environment,
    FindingKind,
    EvidenceRecord,
    EvidenceRequest,
    EvidenceResponse,
    PlannedStep,
    QAState,
    RequestSpec,
    Severity,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_value", ["info", "NONE", None])
@pytest.mark.parametrize("fence", ["{}", "```json\n{}\n```"])
async def test_reflection_normalizes_provider_severity(
    monkeypatch, provider_value, fence
) -> None:
    async def completion(**kwargs):
        body = {
            "result": "pass",
            "interpretation": "Health contract is valid.",
            "adaptation": "Continue with another capability.",
            "continue_testing": True,
            "severity": provider_value,
        }
        return {
            # OpenCode Go models sometimes fence the JSON despite response_format.
            "message": {
                "content": fence.replace("{}", json.dumps(body)),
                "tool_calls": [],
            },
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr("agentic_api_qa.llm._chat_completion", completion)
    step = PlannedStep(
        step_id="health",
        name="Health",
        objective="Validate health",
        request=RequestSpec(operation_id="health", method="GET", path="/api/health"),
        expected_status_codes=[200],
    )
    evidence = EvidenceRecord(
        actor=Actor.EXPLORER,
        iteration=1,
        scenario_or_rule_id="health",
        request=EvidenceRequest(
            method="GET",
            url="https://www.alexpavsky.com/api/health",
            body_sha256="x",
            sanitized_body_excerpt="",
        ),
        response=EvidenceResponse(
            status_code=200,
            selected_headers={"content-type": "application/json"},
            body_sha256="y",
            sanitized_body_excerpt='{"status":"ok"}',
        ),
    )

    result = await reflect_on_evidence(QAState(), Actor.EXPLORER, step, evidence)

    assert result.result == "PASS"
    assert result.severity == Severity.INFO


@pytest.mark.asyncio
async def test_planner_self_corrects_an_incomplete_tool_call(monkeypatch) -> None:
    responses = [
        {
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "bad",
                    "function": {
                        "name": "execute_http_check",
                        "arguments": json.dumps({"step_id": "health"}),
                    },
                }],
            },
            "usage": {"total_tokens": 4},
        },
        {
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "good",
                    "function": {
                        "name": "execute_http_check",
                        "arguments": json.dumps({
                            "step_id": "health",
                            "name": "Health",
                            "objective": "Validate health",
                            "operation_id": "health",
                            "method": "GET",
                            "path": "/api/health",
                            "effect": "read",
                            "expected_status_codes": [200],
                            "decision_summary": "Start with a low-risk contract check.",
                        }),
                    },
                }],
            },
            "usage": {"total_tokens": 6},
        },
    ]

    async def completion(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr("agentic_api_qa.llm._chat_completion", completion)

    choice = await choose_next_action(QAState(), Actor.EXPLORER)

    assert choice.step is not None
    assert choice.step.request.path == "/api/health"
    assert choice.usage.total_tokens == 10


@pytest.mark.asyncio
async def test_judge_gets_its_own_reasoning_budget(monkeypatch) -> None:
    calls: list[dict] = []
    finding = CandidateFinding(
        source="explorer",
        kind=FindingKind.API,
        severity=Severity.HIGH,
        title="Chat API returned HTTP 500",
        expected="A valid chat request returns a reply.",
        actual="POST /api/chat returned an internal server error.",
    )

    async def completion(**kwargs):
        calls.append(kwargs)
        decision = {
            "finding_id": finding.finding_id,
            "verdict": "CONFIRMED",
            "confidence": 0.9,
            "skeptical_challenge": "Could be a harness error.",
            "evidence_for": [],
            "evidence_against": [],
            "reasoning_summary": "Reproducible 500.",
            "investigation_reason": "Product defect.",
            "healer_eligible": False,
        }
        return {
            "message": {"content": json.dumps({"decisions": [decision]}), "tool_calls": []},
            "usage": {"total_tokens": 1},
        }

    monkeypatch.setattr("agentic_api_qa.llm._chat_completion", completion)

    await evaluate_candidate_findings(QAState(), [finding])

    budget = calls[0]["model_config"]
    assert calls[0]["reasoning_effort"] == "high"
    assert budget.max_tokens >= 16_000
    assert budget.timeout_seconds >= 300


@pytest.mark.asyncio
async def test_judge_keeps_returned_verdicts_when_it_skips_a_finding(monkeypatch) -> None:
    def finding(title: str) -> CandidateFinding:
        return CandidateFinding(
            source="explorer",
            kind=FindingKind.API,
            severity=Severity.HIGH,
            title=title,
            expected="A valid chat request returns a reply.",
            actual="POST /api/chat returned an internal server error.",
        )

    judged, skipped = finding("Chat API returned HTTP 500"), finding("Chat reply was empty")

    async def completion(**kwargs):
        decision = {
            "finding_id": judged.finding_id,
            "verdict": "CONFIRMED",
            "confidence": 0.9,
            "skeptical_challenge": "Could be a harness error.",
            "evidence_for": [],
            "evidence_against": [],
            "reasoning_summary": "Reproducible 500.",
            "investigation_reason": "Product defect.",
            "healer_eligible": True,
        }
        return {
            "message": {"content": json.dumps({"decisions": [decision]}), "tool_calls": []},
            "usage": {"total_tokens": 1},
        }

    monkeypatch.setattr("agentic_api_qa.llm._chat_completion", completion)

    result = await evaluate_candidate_findings(QAState(), [judged, skipped])

    verdicts = {item.finding_id: item for item in result.decisions}
    assert verdicts[judged.finding_id].verdict.value == "CONFIRMED"
    assert verdicts[skipped.finding_id].verdict.value == "INCONCLUSIVE"
    assert not verdicts[skipped.finding_id].healer_eligible


@pytest.mark.asyncio
async def test_judge_retries_once_after_malformed_json(monkeypatch) -> None:
    finding = CandidateFinding(
        source="explorer",
        kind=FindingKind.UI,
        severity=Severity.HIGH,
        title="Desktop homepage renders",
        expected="The hero heading is visible.",
        actual="Expected visible text not found.",
    )
    decision = {
        "finding_id": finding.finding_id,
        "verdict": "CONFIRMED",
        "confidence": 0.9,
        "skeptical_challenge": "Could be a slow render.",
        "evidence_for": [],
        "evidence_against": [],
        "reasoning_summary": "Heading is missing.",
        "investigation_reason": "Product defect.",
        "healer_eligible": False,
    }
    replies = ['{"decisions": [', json.dumps({"decisions": [decision]})]

    async def completion(**kwargs):
        return {
            "message": {"content": replies.pop(0), "tool_calls": []},
            "usage": {"total_tokens": 5},
        }

    monkeypatch.setattr("agentic_api_qa.llm._chat_completion", completion)

    result = await evaluate_candidate_findings(QAState(), [finding])

    assert result.decisions[0].verdict.value == "CONFIRMED"
    assert result.usage.total_tokens == 10
