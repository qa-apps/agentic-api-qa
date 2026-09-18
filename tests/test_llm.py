import json

import pytest

from agentic_api_qa.llm import choose_next_action, reflect_on_evidence
from agentic_api_qa.models import (
    Actor,
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
async def test_reflection_normalizes_provider_severity(monkeypatch, provider_value) -> None:
    async def completion(**kwargs):
        body = {
            "result": "pass",
            "interpretation": "Health contract is valid.",
            "adaptation": "Continue with another capability.",
            "continue_testing": True,
            "severity": provider_value,
        }
        return {
            "message": {"content": json.dumps(body), "tool_calls": []},
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
