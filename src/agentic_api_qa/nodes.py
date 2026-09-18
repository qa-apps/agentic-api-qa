"""LangGraph nodes for autonomous, bounded Explorer and Adversary agents."""

from __future__ import annotations

import json

from agentic_api_qa.client import execute_request
from agentic_api_qa.llm import AgentLLMError, UsageDelta, choose_next_action, reflect_on_evidence
from agentic_api_qa.models import (
    Actor,
    AdversarialProposal,
    AgentReflection,
    AuditEvent,
    EvidenceRecord,
    EvidenceRequest,
    EvidenceResponse,
    FinalReport,
    GuardrailDecision,
    GuardrailResult,
    HappyPathProposal,
    HappyPathReport,
    HappyPathResult,
    HappyPathStepDecision,
    ModelUsage,
    OverallResult,
    PipelineError,
    PipelinePhase,
    QAState,
    ReportMetrics,
    ReportRun,
    Severity,
    utc_now,
)
from agentic_api_qa.policy import evaluate_policy


def _audit(
    *,
    actor: Actor,
    event_type: str,
    iteration: int = 0,
    tool_call_count: int = 0,
    scenario_or_rule_id: str | None = None,
    evidence_ids: list[str] | None = None,
    approval_id: str | None = None,
    details: dict | None = None,
) -> AuditEvent:
    return AuditEvent(
        actor=actor,
        event_type=event_type,
        iteration=iteration,
        tool_call_count=tool_call_count,
        scenario_or_rule_id=scenario_or_rule_id,
        evidence_ids=evidence_ids or [],
        approval_id=approval_id,
        redacted_details=details or {},
    )


def _add_usage(current: ModelUsage, delta: UsageDelta) -> ModelUsage:
    return ModelUsage(
        calls=current.calls + 1,
        prompt_tokens=current.prompt_tokens + delta.prompt_tokens,
        completion_tokens=current.completion_tokens + delta.completion_tokens,
        total_tokens=current.total_tokens + delta.total_tokens,
    )


def _synthetic_policy_evidence(
    state: QAState, actor: Actor, step, reason: str
) -> EvidenceRecord:
    body = (
        json.dumps(step.request.json_body, default=str)
        if step.request.json_body is not None
        else ""
    )
    iteration = (
        state.explorer_iterations if actor == Actor.EXPLORER else state.adversary_iterations
    ) + 1
    return EvidenceRecord(
        actor=actor,
        iteration=iteration,
        scenario_or_rule_id=step.rule_id or step.step_id,
        tool_call_id=state.pending_tool_call_id or "policy-blocked",
        request=EvidenceRequest(
            method=step.request.method,
            url=f"{str(state.config.base_url).rstrip('/')}{step.request.path}",
            body_sha256="policy-blocked",
            sanitized_body_excerpt=body[:500],
        ),
        response=EvidenceResponse(transport_error=f"PolicyBlocked: {reason}"),
    )


def _budget_exhausted(state: QAState, actor: Actor) -> bool:
    limits = state.config.limits
    if state.total_tool_calls >= limits.total_tool_calls:
        return True
    if actor == Actor.EXPLORER:
        return (
            state.explorer_iterations >= limits.explorer_iterations
            or state.explorer_tool_calls >= limits.explorer_tool_calls
        )
    return (
        state.adversary_iterations >= limits.adversary_iterations
        or state.adversary_tool_calls >= limits.adversary_tool_calls
    )


def _llm_failure(state: QAState, actor: Actor, exc: Exception) -> dict:
    phase = PipelinePhase.EXPLORER if actor == Actor.EXPLORER else PipelinePhase.ADVERSARY
    error = PipelineError(
        phase=phase,
        code=f"{actor.value.upper()}_LLM_ERROR",
        message=str(exc),
    )
    return {
        f"{actor.value}_done": True,
        f"{actor.value}_stop_reason": "llm_error",
        "pending_step": None,
        "pending_tool_call_id": None,
        "pending_decision_summary": None,
        "pipeline_errors": [*state.pipeline_errors, error],
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=actor,
                event_type="LLM_DECISION_FAILED",
                details={"error": str(exc)},
            ),
        ],
    }


def governor(state: QAState) -> dict:
    return {
        "current_phase": PipelinePhase.EXPLORER,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.GOVERNOR,
                event_type="RUN_INITIALIZED",
                details={
                    "target": state.config.target_name,
                    "environment": state.config.environment.value,
                    "production_read_only": state.config.production_read_only,
                    "explorer_model": state.config.models.explorer_model,
                    "adversary_model": state.config.models.adversary_model,
                },
            ),
        ],
    }


async def _agent_decide(state: QAState, actor: Actor) -> dict:
    done_field = f"{actor.value}_done"
    stop_field = f"{actor.value}_stop_reason"
    if getattr(state, done_field):
        return {}
    if _budget_exhausted(state, actor):
        return {done_field: True, stop_field: "budget_exhausted"}
    try:
        choice = await choose_next_action(state, actor)
    except (AgentLLMError, ValueError, TypeError) as exc:
        return _llm_failure(state, actor, exc)

    plan = state.explorer_plan if actor == Actor.EXPLORER else state.adversary_plan
    iteration = state.explorer_iterations if actor == Actor.EXPLORER else state.adversary_iterations
    calls = state.explorer_tool_calls if actor == Actor.EXPLORER else state.adversary_tool_calls
    event = _audit(
        actor=actor,
        event_type="AGENT_FINISHED" if choice.action == "finish" else "AGENT_SELECTED_TOOL",
        iteration=iteration,
        tool_call_count=calls,
        scenario_or_rule_id=(choice.step.rule_id or choice.step.step_id) if choice.step else None,
        details={
            "decision_summary": choice.summary,
            "tool_name": "execute_http_check" if choice.step else "finish_phase",
            "model": (
                state.config.models.explorer_model
                if actor == Actor.EXPLORER
                else state.config.models.adversary_model
            ),
        },
    )
    updates = {
        "current_phase": (
            PipelinePhase.EXPLORER if actor == Actor.EXPLORER else PipelinePhase.ADVERSARY
        ),
        "model_usage": _add_usage(state.model_usage, choice.usage),
        "audit_log": [*state.audit_log, event],
    }
    if choice.action == "finish":
        return {**updates, done_field: True, stop_field: "agent_finished"}
    assert choice.step is not None
    return {
        **updates,
        f"{actor.value}_plan": [*plan, choice.step],
        "pending_step": choice.step,
        "pending_tool_call_id": choice.tool_call_id,
        "pending_decision_summary": choice.summary,
    }


async def explorer_agent(state: QAState) -> dict:
    return await _agent_decide(state, Actor.EXPLORER)


async def adversary_agent(state: QAState) -> dict:
    return await _agent_decide(state, Actor.ADVERSARY)


def route_after_explorer_agent(state: QAState) -> str:
    return "adversary_agent" if state.explorer_done else "explorer_http_tool"


def route_after_adversary_agent(state: QAState) -> str:
    return "judge" if state.adversary_done else "adversary_http_tool"


async def _http_tool(state: QAState, actor: Actor) -> dict:
    step = state.pending_step
    if step is None:
        raise RuntimeError(f"{actor.value} HTTP tool invoked without a pending step")
    iteration = (
        state.explorer_iterations + 1
        if actor == Actor.EXPLORER
        else state.adversary_iterations + 1
    )
    policy = evaluate_policy(state.config, step.request, state.approvals)
    evidence = (
        await execute_request(
            config=state.config,
            request=step.request,
            actor=actor,
            iteration=iteration,
            scenario_or_rule_id=step.rule_id or step.step_id,
        )
        if policy.allowed
        else _synthetic_policy_evidence(state, actor, step, policy.reason)
    )
    if state.pending_tool_call_id:
        evidence = evidence.model_copy(update={"tool_call_id": state.pending_tool_call_id})
    old_calls = state.explorer_tool_calls if actor == Actor.EXPLORER else state.adversary_tool_calls
    calls = old_calls + int(policy.allowed)
    return {
        f"{actor.value}_iterations": iteration,
        f"{actor.value}_tool_calls": calls,
        "total_tool_calls": state.total_tool_calls + int(policy.allowed),
        "last_evidence_id": evidence.evidence_id,
        "evidence": {**state.evidence, evidence.evidence_id: evidence},
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=actor,
                event_type="TOOL_CALL_COMPLETED" if policy.allowed else "POLICY_BLOCKED",
                iteration=iteration,
                tool_call_count=calls,
                scenario_or_rule_id=step.rule_id or step.step_id,
                evidence_ids=[evidence.evidence_id],
                approval_id=policy.approval_id,
                details={
                    "policy_reason": policy.reason,
                    "method": step.request.method,
                    "path": step.request.path,
                },
            ),
        ],
    }


async def explorer_http_tool(state: QAState) -> dict:
    return await _http_tool(state, Actor.EXPLORER)


async def adversary_http_tool(state: QAState) -> dict:
    return await _http_tool(state, Actor.ADVERSARY)


async def _observe(state: QAState, actor: Actor) -> dict:
    step = state.pending_step
    evidence = state.evidence.get(state.last_evidence_id or "")
    if step is None or evidence is None:
        return _llm_failure(
            state, actor, AgentLLMError(f"{actor.value} observation lacks step evidence")
        )
    try:
        analysis = await reflect_on_evidence(state, actor, step, evidence)
        result = (
            HappyPathResult(analysis.result)
            if actor == Actor.EXPLORER
            else GuardrailResult(analysis.result)
        )
    except (AgentLLMError, ValueError, TypeError) as exc:
        return _llm_failure(state, actor, exc)

    reflection = AgentReflection(
        actor=actor,
        iteration=(
            state.explorer_iterations if actor == Actor.EXPLORER else state.adversary_iterations
        ),
        scenario_or_rule_id=step.rule_id or step.step_id,
        result=result.value,
        interpretation=analysis.interpretation,
        adaptation=analysis.adaptation,
        continue_testing=analysis.continue_testing,
        evidence_id=evidence.evidence_id,
    )
    updates: dict = {
        f"{actor.value}_index": (
            state.explorer_index + 1 if actor == Actor.EXPLORER else state.adversary_index + 1
        ),
        f"{actor.value}_reflections": [
            *(
                state.explorer_reflections
                if actor == Actor.EXPLORER
                else state.adversary_reflections
            ),
            reflection,
        ],
        f"{actor.value}_done": not analysis.continue_testing,
        f"{actor.value}_stop_reason": (
            "observer_finished" if not analysis.continue_testing else None
        ),
        "pending_step": None,
        "pending_tool_call_id": None,
        "pending_decision_summary": None,
        "model_usage": _add_usage(state.model_usage, analysis.usage),
    }
    if actor == Actor.EXPLORER:
        proposal = HappyPathProposal(
            step_id=step.step_id,
            proposed_result=result,
            interpretation=analysis.interpretation,
            evidence_ids=[evidence.evidence_id],
        )
        updates["explorer_proposals"] = [*state.explorer_proposals, proposal]
    else:
        proposal = AdversarialProposal(
            rule_id=step.rule_id or step.step_id,
            name=step.name,
            attack_class=step.attack_class or "UNKNOWN",
            proposed_result=result,
            severity=analysis.severity,
            interpretation=analysis.interpretation,
            evidence_ids=[evidence.evidence_id],
        )
        updates["adversary_proposals"] = [*state.adversary_proposals, proposal]
    updates["audit_log"] = [
        *state.audit_log,
        _audit(
            actor=actor,
            event_type="AGENT_REFLECTED",
            iteration=reflection.iteration,
            tool_call_count=(
                state.explorer_tool_calls if actor == Actor.EXPLORER else state.adversary_tool_calls
            ),
            scenario_or_rule_id=reflection.scenario_or_rule_id,
            evidence_ids=[evidence.evidence_id],
            details={
                "result": result.value,
                "severity": analysis.severity.value,
                "adaptation": analysis.adaptation,
                "continue_testing": analysis.continue_testing,
            },
        ),
    ]
    return updates


async def explorer_observe(state: QAState) -> dict:
    return await _observe(state, Actor.EXPLORER)


async def adversary_observe(state: QAState) -> dict:
    return await _observe(state, Actor.ADVERSARY)


def route_after_explorer_observe(state: QAState) -> str:
    return "adversary_agent" if state.explorer_done else "explorer_agent"


def route_after_adversary_observe(state: QAState) -> str:
    return "judge" if state.adversary_done else "adversary_agent"


def _has_phase_error(state: QAState, phase: PipelinePhase) -> bool:
    return any(error.phase == phase for error in state.pipeline_errors)


def judge(state: QAState) -> dict:
    happy_by_id = {step.step_id: step for step in state.explorer_plan}
    happy_decisions = [
        HappyPathStepDecision(
            step_id=item.step_id,
            name=happy_by_id[item.step_id].name,
            result=item.proposed_result,
            reason=item.interpretation,
            evidence_ids=item.evidence_ids,
        )
        for item in state.explorer_proposals
    ]
    guardrail_decisions = [
        GuardrailDecision(
            rule_id=item.rule_id,
            name=item.name,
            attack_class=item.attack_class,
            result=item.proposed_result,
            severity=item.severity,
            reason=item.interpretation,
            evidence_ids=item.evidence_ids,
        )
        for item in state.adversary_proposals
    ]
    errors = list(state.pipeline_errors)
    if not happy_decisions and not _has_phase_error(state, PipelinePhase.EXPLORER):
        errors.append(PipelineError(
            phase=PipelinePhase.EXPLORER,
            code="EXPLORER_NO_EVIDENCE",
            message="Explorer finished without producing test evidence.",
        ))
    elif state.explorer_stop_reason == "budget_exhausted":
        errors.append(PipelineError(
            phase=PipelinePhase.EXPLORER,
            code="EXPLORER_BUDGET_EXHAUSTED",
            message="Explorer reached its safety budget before choosing to finish.",
        ))
    if not guardrail_decisions and not _has_phase_error(state, PipelinePhase.ADVERSARY):
        errors.append(PipelineError(
            phase=PipelinePhase.ADVERSARY,
            code="ADVERSARY_NO_EVIDENCE",
            message="Adversary finished without producing test evidence.",
        ))
    elif state.adversary_stop_reason == "budget_exhausted":
        errors.append(PipelineError(
            phase=PipelinePhase.ADVERSARY,
            code="ADVERSARY_BUDGET_EXHAUSTED",
            message="Adversary reached its safety budget before choosing to finish.",
        ))
    return {
        "current_phase": PipelinePhase.JUDGE,
        "happy_path_decisions": happy_decisions,
        "guardrail_decisions": guardrail_decisions,
        "pipeline_errors": errors,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.JUDGE,
                event_type="DECISIONS_FINALIZED",
                details={
                    "happy_path_decisions": len(happy_decisions),
                    "guardrail_decisions": len(guardrail_decisions),
                    "pipeline_errors": len(errors),
                },
            ),
        ],
    }


def reporter(state: QAState) -> dict:
    finished_at = utc_now()
    happy_failures = sum(item.result == HappyPathResult.FAIL for item in state.happy_path_decisions)
    breaches = sum(item.result == GuardrailResult.BREACHED for item in state.guardrail_decisions)
    inconclusive = sum(item.result == GuardrailResult.INCONCLUSIVE for item in state.guardrail_decisions)
    if any(error.blocking for error in state.pipeline_errors) or inconclusive:
        overall = OverallResult.ERROR
    elif happy_failures or breaches:
        overall = OverallResult.FAIL
    else:
        overall = OverallResult.PASS
    metrics = ReportMetrics(
        happy_path_passed=len(state.happy_path_decisions) - happy_failures,
        happy_path_failed=happy_failures,
        guardrails_held=sum(item.result == GuardrailResult.HELD for item in state.guardrail_decisions),
        guardrails_breached=breaches,
        guardrails_inconclusive=inconclusive,
        pipeline_errors=len(state.pipeline_errors),
        explorer_iterations=state.explorer_iterations,
        adversary_iterations=state.adversary_iterations,
        explorer_tool_calls=state.explorer_tool_calls,
        adversary_tool_calls=state.adversary_tool_calls,
        total_tool_calls=state.total_tool_calls,
        llm_calls=state.model_usage.calls,
        llm_prompt_tokens=state.model_usage.prompt_tokens,
        llm_completion_tokens=state.model_usage.completion_tokens,
        llm_total_tokens=state.model_usage.total_tokens,
    )
    summary = (
        f"Happy path: {metrics.happy_path_passed} passed, {metrics.happy_path_failed} failed. "
        f"Guardrails: {metrics.guardrails_held} held, {metrics.guardrails_breached} breached, "
        f"{metrics.guardrails_inconclusive} inconclusive. LLM calls: {metrics.llm_calls}."
    )
    audit = [
        *state.audit_log,
        _audit(actor=Actor.REPORTER, event_type="REPORT_GENERATED", details={"overall_result": overall.value}),
    ]
    report = FinalReport(
        run=ReportRun(
            run_id=state.run_id,
            target=str(state.config.base_url),
            environment=state.config.environment,
            started_at=state.started_at,
            finished_at=finished_at,
            duration_ms=max(0, (finished_at - state.started_at).total_seconds() * 1_000),
            overall_result=overall,
            production_read_only=state.config.production_read_only,
        ),
        happy_path=HappyPathReport(
            status=HappyPathResult.FAIL if happy_failures else HappyPathResult.PASS,
            steps=state.happy_path_decisions,
        ),
        adversarial=state.guardrail_decisions,
        summary=summary,
        metrics=metrics,
        evidence=list(state.evidence.values()),
        audit_log=audit,
        pipeline_errors=state.pipeline_errors,
    )
    return {
        "finished_at": finished_at,
        "current_phase": PipelinePhase.COMPLETE,
        "audit_log": audit,
        "report": report,
    }
