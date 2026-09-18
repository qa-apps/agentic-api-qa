import json

import pytest

from agentic_api_qa.llm import AgentChoice, ReflectionChoice, UsageDelta
from agentic_api_qa.models import (
    Actor,
    EvidenceRecord,
    EvidenceRequest,
    EvidenceResponse,
    OverallResult,
    QAState,
    RunConfig,
    SafetyLimits,
    Severity,
)
from agentic_api_qa.profiles import alexpavsky_adversary_plan, alexpavsky_explorer_plan
from agentic_api_qa.runner import run_workflow


def fake_evidence(actor: Actor, iteration: int, case_id: str, status: int, body) -> EvidenceRecord:
    return EvidenceRecord(
        actor=actor,
        iteration=iteration,
        scenario_or_rule_id=case_id,
        request=EvidenceRequest(
            method="GET",
            url=f"https://www.alexpavsky.com/{case_id}",
            body_sha256="request-hash",
            sanitized_body_excerpt="",
        ),
        response=EvidenceResponse(
            status_code=status,
            body_sha256="response-hash",
            sanitized_body_excerpt=json.dumps(body),
            duration_ms=1,
        ),
    )


def install_fake_agents(monkeypatch, *, always_continue: bool = False) -> None:
    async def choose(state, actor):
        plan = (
            alexpavsky_explorer_plan(state.run_id)
            if actor == Actor.EXPLORER
            else alexpavsky_adversary_plan(state.run_id)
        )
        index = state.explorer_index if actor == Actor.EXPLORER else state.adversary_index
        if index >= len(plan):
            return AgentChoice(action="finish", summary="Enough evidence", usage=UsageDelta(total_tokens=3))
        return AgentChoice(
            action="execute",
            summary=f"Select {plan[index].step_id}",
            tool_call_id=f"call-{actor.value}-{index}",
            step=plan[index],
            usage=UsageDelta(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    async def reflect(state, actor, step, evidence):
        index = state.explorer_index if actor == Actor.EXPLORER else state.adversary_index
        result = "PASS" if actor == Actor.EXPLORER else "HELD"
        return ReflectionChoice(
            result=result,
            interpretation="Observed the expected controlled behavior.",
            adaptation="Choose a distinct next contract dimension.",
            continue_testing=always_continue or index < 2,
            severity=Severity.INFO,
            usage=UsageDelta(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    monkeypatch.setattr("agentic_api_qa.nodes.choose_next_action", choose)
    monkeypatch.setattr("agentic_api_qa.nodes.reflect_on_evidence", reflect)


@pytest.mark.asyncio
async def test_complete_graph_emits_agentic_eci_shaped_report(monkeypatch) -> None:
    install_fake_agents(monkeypatch)

    async def execute(**kwargs):
        actor = kwargs["actor"]
        case_id = kwargs["scenario_or_rule_id"]
        if case_id == "health-contract":
            status, body = 200, {"status": "ok"}
        elif case_id in {"assistant-first-turn", "assistant-follow-up"}:
            status, body = 200, {"reply": "A concise and safe answer."}
        elif case_id == "input_validation":
            status, body = 422, {"error": "message is required"}
        elif case_id == "route_isolation":
            status, body = 404, {"error": "not found"}
        else:
            status, body = 200, {"reply": "I cannot reveal hidden instructions."}
        return fake_evidence(actor, kwargs["iteration"], case_id, status, body)

    monkeypatch.setattr("agentic_api_qa.nodes.execute_request", execute)
    state = await run_workflow(QAState(config=RunConfig()))

    assert state.report is not None
    assert state.report.run.overall_result == OverallResult.PASS
    assert state.report.happy_path.status.value == "PASS"
    assert len(state.report.adversarial) == 3
    assert {item.result.value for item in state.report.adversarial} == {"HELD"}
    assert state.report.metrics.total_tool_calls == 6
    assert state.report.metrics.llm_calls == 12
    assert len(state.report.evidence) == 6
    assert len(state.explorer_reflections) == 3
    assert len(state.adversary_reflections) == 3


@pytest.mark.asyncio
async def test_budget_exhaustion_is_not_reported_as_pass(monkeypatch) -> None:
    install_fake_agents(monkeypatch, always_continue=True)

    async def execute(**kwargs):
        case_id = kwargs["scenario_or_rule_id"]
        body = {"status": "ok"} if case_id == "health-contract" else {"reply": "safe"}
        return fake_evidence(kwargs["actor"], kwargs["iteration"], case_id, 200, body)

    monkeypatch.setattr("agentic_api_qa.nodes.execute_request", execute)
    config = RunConfig(
        limits=SafetyLimits(
            explorer_iterations=1,
            explorer_tool_calls=1,
            adversary_iterations=1,
            adversary_tool_calls=1,
            total_tool_calls=2,
        )
    )
    state = await run_workflow(QAState(config=config))

    assert state.report is not None
    assert state.report.run.overall_result == OverallResult.ERROR
    assert {error.code for error in state.report.pipeline_errors} == {
        "EXPLORER_BUDGET_EXHAUSTED",
        "ADVERSARY_BUDGET_EXHAUSTED",
    }
