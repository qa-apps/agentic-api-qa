"""LangGraph nodes for autonomous, bounded Explorer and Adversary agents."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_api_qa.client import execute_request
from agentic_api_qa.llm import (
    AgentLLMError,
    UsageDelta,
    choose_next_ui_case,
    choose_next_action,
    evaluate_ui_design,
    evaluate_candidate_findings,
    review_healer_pull_request,
    reflect_on_evidence,
)
from agentic_api_qa.healer import (
    HealerError,
    apply_validated_patch,
    capture_after_ui_evidence,
    plan_confirmed_fix,
    plan_reviewer_revision,
    publish_draft_pr,
    run_validation_profiles,
)
from agentic_api_qa.human_review import notify_human_review
from agentic_api_qa.reviewer import (
    ReviewerError,
    current_commit,
    publish_reviewer_comment,
    pull_request_diff,
    run_reviewer_scans,
)
from agentic_api_qa.mcp_browser import execute_ui_smoke_case
from agentic_api_qa.observability import observe_node
from agentic_api_qa.models import (
    Actor,
    AdversarialProposal,
    AgentReflection,
    AuditEvent,
    EvidenceRecord,
    EvidenceRequest,
    EvidenceResponse,
    FinalReport,
    CandidateFinding,
    FindingKind,
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
    JudgeDecision,
    JudgeVerdict,
    HealerStatus,
    ReviewerVerdict,
    UICaseResult,
    utc_now,
)
from agentic_api_qa.policy import evaluate_policy
from agentic_api_qa.reporting import publish_report_bundle
from agentic_api_qa.report_server import ensure_report_server
from agentic_api_qa.ui_catalog import alexpavsky_ui_smoke_catalog


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
    iterations = (
        state.explorer_iterations if actor == Actor.EXPLORER else state.adversary_iterations
    )
    tool_calls = (
        state.explorer_tool_calls if actor == Actor.EXPLORER else state.adversary_tool_calls
    )
    configured_iterations = (
        limits.explorer_iterations if actor == Actor.EXPLORER else limits.adversary_iterations
    )
    configured_calls = (
        limits.explorer_tool_calls if actor == Actor.EXPLORER else limits.adversary_tool_calls
    )
    if iterations >= configured_iterations or tool_calls >= configured_calls:
        return True
    if iterations >= limits.hard_case_limit:
        return True
    if iterations >= limits.soft_case_limit and not _serious_api_bug_found(state, actor):
        return True
    return False


def _serious_api_bug_ids(state: QAState, actor: Actor) -> list[str]:
    if actor == Actor.EXPLORER:
        return [
            evidence.scenario_or_rule_id
            for evidence in state.evidence.values()
            if evidence.actor == actor
            and (
                evidence.response.transport_error is not None
                or (evidence.response.status_code or 0) >= 500
            )
        ]
    return [
        item.rule_id
        for item in state.adversary_proposals
        if item.proposed_result == GuardrailResult.BREACHED
        and item.severity in {Severity.HIGH, Severity.CRITICAL}
    ]


def _serious_api_bug_found(state: QAState, actor: Actor) -> bool:
    return bool(_serious_api_bug_ids(state, actor))


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


@observe_node(name="safety_governor", actor="governor", as_type="guardrail")
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

    current_count = (
        state.explorer_iterations if actor == Actor.EXPLORER else state.adversary_iterations
    )
    if choice.action == "execute" and current_count >= state.config.limits.soft_case_limit:
        serious_ids = _serious_api_bug_ids(state, actor)
        cited = any(
            case_id in (choice.investigation_reason or "") for case_id in serious_ids
        )
        if not serious_ids or not choice.investigation_reason or not cited:
            return {
                done_field: True,
                stop_field: "soft_limit_reached_without_investigation_justification",
                "pending_step": None,
                "pending_tool_call_id": None,
                "pending_decision_summary": None,
                "model_usage": _add_usage(state.model_usage, choice.usage),
                "audit_log": [
                    *state.audit_log,
                    _audit(
                        actor=actor,
                        event_type="SOFT_LIMIT_STOPPED_UNJUSTIFIED_EXPANSION",
                        iteration=current_count,
                        details={"serious_bug_case_ids": serious_ids},
                    ),
                ],
            }

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
            "investigation_reason": choice.investigation_reason,
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


@observe_node(name="explorer_agent", actor="explorer", as_type="agent")
async def explorer_agent(state: QAState) -> dict:
    return await _agent_decide(state, Actor.EXPLORER)


@observe_node(name="adversary_agent", actor="adversary", as_type="agent")
async def adversary_agent(state: QAState) -> dict:
    return await _agent_decide(state, Actor.ADVERSARY)


def route_after_explorer_agent(state: QAState) -> str:
    return "adversary_agent" if state.explorer_done else "explorer_http_tool"


def route_after_adversary_agent(state: QAState) -> str:
    return "ui_explorer_agent" if state.adversary_done else "adversary_http_tool"


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


@observe_node(name="explorer_http_tool", actor="explorer", as_type="tool")
async def explorer_http_tool(state: QAState) -> dict:
    return await _http_tool(state, Actor.EXPLORER)


@observe_node(name="adversary_http_tool", actor="adversary", as_type="tool")
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


@observe_node(name="explorer_observe", actor="explorer", as_type="span")
async def explorer_observe(state: QAState) -> dict:
    return await _observe(state, Actor.EXPLORER)


@observe_node(name="adversary_observe", actor="adversary", as_type="span")
async def adversary_observe(state: QAState) -> dict:
    return await _observe(state, Actor.ADVERSARY)


def route_after_explorer_observe(state: QAState) -> str:
    return "adversary_agent" if state.explorer_done else "explorer_agent"


def route_after_adversary_observe(state: QAState) -> str:
    return "ui_explorer_agent" if state.adversary_done else "adversary_agent"


@observe_node(name="ui_explorer_agent", actor="ui_explorer", as_type="agent")
async def ui_explorer_agent(state: QAState) -> dict:
    """Choose one UI case at a time from fresh evidence and bounded candidates."""

    if not state.config.ui_enabled:
        return {
            "current_phase": PipelinePhase.UI_EXPLORER,
            "ui_plan_summary": "UI checks disabled by configuration.",
            "ui_done": True,
            "ui_stop_reason": "disabled",
        }
    catalog = alexpavsky_ui_smoke_catalog()
    by_id = {case.case_id: case for case in catalog}
    executed_ids = {item.case_id for item in state.ui_results}
    remaining = [case for case in catalog if case.case_id not in executed_ids]
    count = len(state.ui_results)
    limits = state.config.limits
    if count >= limits.hard_case_limit:
        return {
            "ui_done": True,
            "ui_stop_reason": "hard_limit_reached",
            "ui_pending_case": None,
            "ui_skipped_case_ids": [case.case_id for case in remaining],
        }
    if count >= limits.soft_case_limit and not state.ui_serious_bug_case_ids:
        return {
            "ui_done": True,
            "ui_stop_reason": "soft_limit_reached_without_serious_bug",
            "ui_pending_case": None,
            "ui_skipped_case_ids": [case.case_id for case in remaining],
        }
    if not remaining:
        return {
            "ui_done": True,
            "ui_stop_reason": "catalog_exhausted",
            "ui_pending_case": None,
        }

    error = None
    usage = UsageDelta()
    try:
        choice = await choose_next_ui_case(state, catalog)
        summary = choice.summary
        usage = choice.usage
    except (AgentLLMError, ValueError, TypeError) as exc:
        # Below the soft limit a deterministic next case preserves useful evidence.
        # Above it, never expand scope when the agent cannot justify investigation.
        choice = None
        summary = "UI decision failed; safe fallback applied."
        error = PipelineError(
            phase=PipelinePhase.UI_EXPLORER,
            code="UI_DECISION_LLM_ERROR",
            message=str(exc),
            recoverable=True,
            blocking=False,
        )
    if choice is None:
        selected = remaining[0] if count < limits.soft_case_limit else None
        action = "execute" if selected is not None else "finish"
        investigation_reason = None
    else:
        action = choice.action
        selected = by_id.get(choice.case_id or "") if action == "execute" else None
        investigation_reason = choice.investigation_reason

    if action == "execute" and count >= limits.soft_case_limit:
        cited = any(
            case_id in (investigation_reason or "")
            for case_id in state.ui_serious_bug_case_ids
        )
        if not state.ui_serious_bug_case_ids or not investigation_reason or not cited:
            action = "finish"
            selected = None
            summary = "Stopped at soft limit: no evidence-linked serious-bug investigation justification."

    if action == "finish" or selected is None:
        return {
            "current_phase": PipelinePhase.UI_EXPLORER,
            "ui_done": True,
            "ui_stop_reason": "agent_finished",
            "ui_pending_case": None,
            "ui_plan_summary": summary,
            "ui_skipped_case_ids": [case.case_id for case in remaining],
            "model_usage": _add_usage(state.model_usage, usage) if usage.total_tokens else state.model_usage,
            "pipeline_errors": [*state.pipeline_errors, *([error] if error else [])],
            "audit_log": [
                *state.audit_log,
                _audit(
                    actor=Actor.UI_EXPLORER,
                    event_type="UI_AGENT_FINISHED",
                    iteration=count,
                    details={"summary": summary, "skipped": len(remaining)},
                ),
            ],
        }

    return {
        "current_phase": PipelinePhase.UI_EXPLORER,
        "ui_plan": [*state.ui_plan, selected],
        "ui_pending_case": selected,
        "ui_plan_summary": summary,
        "ui_investigation_reasons": [
            *state.ui_investigation_reasons,
            *([investigation_reason] if investigation_reason else []),
        ],
        "model_usage": _add_usage(state.model_usage, usage) if usage.total_tokens else state.model_usage,
        "pipeline_errors": [*state.pipeline_errors, *([error] if error else [])],
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.UI_EXPLORER,
                event_type="UI_CASE_SELECTED",
                iteration=count + 1,
                scenario_or_rule_id=selected.case_id,
                details={
                    "summary": summary,
                    "beyond_soft_limit": count >= limits.soft_case_limit,
                    "investigation_reason": investigation_reason,
                },
            ),
        ],
    }


def route_after_ui_explorer(state: QAState) -> str:
    return "design_evaluator" if state.ui_done else "playwright_mcp_tool"


@observe_node(name="playwright_mcp_tool", actor="ui_explorer", as_type="tool")
async def playwright_mcp_tool(state: QAState) -> dict:
    """Execute exactly one agent-selected UI case through Playwright MCP."""

    if not state.config.ui_enabled:
        return {}
    case = state.ui_pending_case
    if case is None:
        return {
            "pipeline_errors": [
                *state.pipeline_errors,
                PipelineError(
                    phase=PipelinePhase.UI_EXPLORER,
                    code="UI_CASE_MISSING",
                    message="Playwright MCP node was invoked without a selected UI case.",
                ),
            ],
            "ui_done": True,
        }
    try:
        result, calls = await execute_ui_smoke_case(
            run_config=state.config,
            run_id=state.run_id,
            case=case,
            calls_so_far=state.ui_mcp_tool_calls,
        )
        errors = state.pipeline_errors
    except Exception as exc:
        result = UICaseResult(
            case_id=case.case_id,
            name=case.name,
            category=case.category,
            passed=False,
            reason=f"Playwright MCP infrastructure failure: {type(exc).__name__}: {exc}",
            severity=Severity.CRITICAL,
            investigation_required=True,
        )
        calls = 0
        errors = [
            *state.pipeline_errors,
            PipelineError(
                phase=PipelinePhase.UI_EXPLORER,
                code="PLAYWRIGHT_MCP_ERROR",
                message=f"{type(exc).__name__}: {exc}",
            ),
        ]
    return {
        "ui_results": [*state.ui_results, result],
        "ui_mcp_tool_calls": state.ui_mcp_tool_calls + calls,
        "pipeline_errors": errors,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.UI_EXPLORER,
                event_type="PLAYWRIGHT_MCP_CASE_COMPLETED",
                iteration=len(state.ui_results) + 1,
                tool_call_count=state.ui_mcp_tool_calls + calls,
                scenario_or_rule_id=case.case_id,
                details={
                    "passed": result.passed,
                    "severity": result.severity.value,
                    "screenshot_captured": result.screenshot_captured,
                    "screenshot_path": result.screenshot_path,
                },
            ),
        ],
    }


@observe_node(name="ui_observe", actor="ui_explorer", as_type="span")
def ui_observe(state: QAState) -> dict:
    """Turn raw MCP results into an auditable UI phase observation."""

    result = state.ui_results[-1] if state.ui_results else None
    serious = list(state.ui_serious_bug_case_ids)
    if result and result.investigation_required and result.case_id not in serious:
        serious.append(result.case_id)
    return {
        "ui_pending_case": None,
        "ui_serious_bug_case_ids": serious,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.UI_EXPLORER,
                event_type="UI_EVIDENCE_OBSERVED",
                tool_call_count=state.ui_mcp_tool_calls,
                iteration=len(state.ui_results),
                scenario_or_rule_id=result.case_id if result else None,
                details={
                    "passed": result.passed if result else None,
                    "severity": result.severity.value if result else None,
                    "serious_bug_case_ids": serious,
                    "next_action": "return_to_ui_agent",
                },
            ),
        ]
    }


def route_after_ui_observe(state: QAState) -> str:
    return "ui_explorer_agent"


@observe_node(name="design_evaluator", actor="design_evaluator", as_type="evaluator")
async def design_evaluator(state: QAState) -> dict:
    """Use a vision-capable LLM judge on screenshots from selected UI states."""

    if not state.config.ui_enabled or not state.ui_results:
        return {"current_phase": PipelinePhase.DESIGN_EVALUATOR}
    try:
        choice = await evaluate_ui_design(state, state.ui_results)
        evaluation = choice.evaluation
        usage = _add_usage(state.model_usage, choice.usage)
        errors = state.pipeline_errors
    except (AgentLLMError, ValueError, TypeError) as exc:
        evaluation = None
        usage = state.model_usage
        errors = [
            *state.pipeline_errors,
            PipelineError(
                phase=PipelinePhase.DESIGN_EVALUATOR,
                code="DESIGN_EVALUATOR_ERROR",
                message=str(exc),
                recoverable=True,
                blocking=False,
            ),
        ]
    return {
        "current_phase": PipelinePhase.DESIGN_EVALUATOR,
        "design_evaluation": evaluation,
        "model_usage": usage,
        "pipeline_errors": errors,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.DESIGN_EVALUATOR,
                event_type="DESIGN_EVALUATED" if evaluation else "DESIGN_EVALUATION_FAILED",
                details={
                    "screenshots": len(
                        evaluation.screenshot_paths if evaluation is not None else []
                    )
                },
            ),
        ],
    }


def _has_phase_error(state: QAState, phase: PipelinePhase) -> bool:
    return any(error.phase == phase for error in state.pipeline_errors)


def _candidate_findings(state: QAState) -> list[CandidateFinding]:
    happy_by_id = {step.step_id: step for step in state.explorer_plan}
    findings: list[CandidateFinding] = []
    for proposal in state.explorer_proposals:
        if proposal.proposed_result != HappyPathResult.FAIL:
            continue
        step = happy_by_id.get(proposal.step_id)
        api_evidence = {
            evidence_id: state.evidence[evidence_id].response.model_dump(mode="json")
            for evidence_id in proposal.evidence_ids
            if evidence_id in state.evidence
        }
        findings.append(
            CandidateFinding(
                source=Actor.EXPLORER,
                kind=FindingKind.API,
                severity=Severity.HIGH,
                title=step.name if step else proposal.step_id,
                expected=(
                    step.semantic_expectation
                    if step and step.semantic_expectation
                    else f"Expected status codes: {step.expected_status_codes if step else 'contract success'}"
                ),
                actual=proposal.interpretation,
                reproduction_steps=[
                    f"Execute Explorer step {proposal.step_id}",
                    "Compare the sanitized response with the declared semantic contract.",
                ],
                evidence_ids=proposal.evidence_ids,
                api_evidence=api_evidence,
            )
        )
    for proposal in state.adversary_proposals:
        if proposal.proposed_result != GuardrailResult.BREACHED:
            continue
        api_evidence = {
            evidence_id: state.evidence[evidence_id].response.model_dump(mode="json")
            for evidence_id in proposal.evidence_ids
            if evidence_id in state.evidence
        }
        findings.append(
            CandidateFinding(
                source=Actor.ADVERSARY,
                kind=FindingKind.API,
                severity=proposal.severity,
                title=proposal.name,
                expected=f"Guardrail {proposal.rule_id} must hold for {proposal.attack_class}.",
                actual=proposal.interpretation,
                reproduction_steps=[
                    f"Execute adversarial rule {proposal.rule_id}",
                    "Confirm that unsafe behavior is observable in the sanitized response.",
                ],
                evidence_ids=proposal.evidence_ids,
                api_evidence=api_evidence,
            )
        )
    for result in state.ui_results:
        if result.passed:
            continue
        findings.append(
            CandidateFinding(
                source=Actor.UI_EXPLORER,
                kind=FindingKind.UI,
                severity=result.severity,
                title=result.name,
                expected=f"UI smoke case {result.case_id} should pass without console errors.",
                actual=result.reason,
                reproduction_steps=[
                    f"Run Playwright MCP case {result.case_id}",
                    "Inspect the full-page screenshot, accessibility snapshot, and console evidence.",
                ],
                screenshot_paths=[result.screenshot_path] if result.screenshot_path else [],
                api_evidence={
                    "ui_case_id": result.case_id,
                    "console_errors": result.console_errors,
                    "mcp_tools": result.mcp_tools,
                    "snapshot_excerpt": result.snapshot_excerpt,
                },
            )
        )
    if state.design_evaluation is not None:
        for index, issue in enumerate(state.design_evaluation.issues[:3], start=1):
            findings.append(
                CandidateFinding(
                    source=Actor.DESIGN_EVALUATOR,
                    kind=FindingKind.DESIGN,
                    severity=Severity.LOW,
                    title=f"Design observation {index}",
                    expected="The UI should satisfy the explicit design rubric.",
                    actual=issue,
                    reproduction_steps=["Review the captured design-checkpoint screenshots."],
                    screenshot_paths=state.design_evaluation.screenshot_paths,
                    design_only=True,
                )
            )
    return findings


def _raw_decisions(state: QAState) -> tuple[list[HappyPathStepDecision], list[GuardrailDecision]]:
    happy_by_id = {step.step_id: step for step in state.explorer_plan}
    happy = [
        HappyPathStepDecision(
            step_id=item.step_id,
            name=happy_by_id[item.step_id].name,
            result=item.proposed_result,
            reason=item.interpretation,
            evidence_ids=item.evidence_ids,
        )
        for item in state.explorer_proposals
        if item.step_id in happy_by_id
    ]
    guardrails = [
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
    return happy, guardrails


@observe_node(name="judge", actor="judge", as_type="evaluator")
async def judge(state: QAState) -> dict:
    happy_decisions, guardrail_decisions = _raw_decisions(state)
    candidates = _candidate_findings(state)
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
    decisions: list[JudgeDecision] = []
    usage = state.model_usage
    if candidates:
        try:
            evaluation = await evaluate_candidate_findings(state, candidates)
            decisions = evaluation.decisions
            usage = _add_usage(state.model_usage, evaluation.usage)
        except (AgentLLMError, ValueError, TypeError) as exc:
            decisions = [
                JudgeDecision(
                    finding_id=finding.finding_id,
                    verdict=JudgeVerdict.INCONCLUSIVE,
                    confidence=0,
                    skeptical_challenge="Independent falsification could not be completed.",
                    evidence_for=[],
                    evidence_against=["Judge model did not return a valid complete evaluation."],
                    reasoning_summary="Fail-closed: no repair is allowed without an independent judgment.",
                    investigation_reason=str(exc),
                    healer_eligible=False,
                )
                for finding in candidates
            ]
            errors.append(
                PipelineError(
                    phase=PipelinePhase.JUDGE,
                    code="JUDGE_LLM_ERROR",
                    message=str(exc),
                    recoverable=True,
                    blocking=True,
                )
            )
    confirmed_ids = {
        decision.finding_id
        for decision in decisions
        if decision.verdict == JudgeVerdict.CONFIRMED
    }
    confirmed = [
        finding for finding in candidates if finding.finding_id in confirmed_ids
    ]
    return {
        "current_phase": PipelinePhase.JUDGE,
        "happy_path_decisions": happy_decisions,
        "guardrail_decisions": guardrail_decisions,
        "candidate_findings": candidates,
        "judge_decisions": decisions,
        "confirmed_findings": confirmed,
        "model_usage": usage,
        "pipeline_errors": errors,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.JUDGE,
                event_type="INDEPENDENT_DECISIONS_FINALIZED",
                details={
                    "happy_path_decisions": len(happy_decisions),
                    "guardrail_decisions": len(guardrail_decisions),
                    "pipeline_errors": len(errors),
                    "ui_cases": len(state.ui_results),
                    "ui_failures": sum(not item.passed for item in state.ui_results),
                    "candidate_findings": len(candidates),
                    "confirmed_findings": len(confirmed),
                    "rejected_findings": sum(
                        item.verdict == JudgeVerdict.REJECTED for item in decisions
                    ),
                    "inconclusive_findings": sum(
                        item.verdict == JudgeVerdict.INCONCLUSIVE for item in decisions
                    ),
                    "model": state.config.models.judge_model,
                },
            ),
        ],
    }


def route_after_judge(state: QAState) -> str:
    if not state.confirmed_findings:
        return "json_reporter"
    eligible = any(item.healer_eligible for item in state.judge_decisions)
    return "healer_agent" if eligible else "human_review_notifier"


def _selected_healer_pair(
    state: QAState,
) -> tuple[CandidateFinding, JudgeDecision] | None:
    severity_rank = {
        Severity.CRITICAL: 4,
        Severity.HIGH: 3,
        Severity.MEDIUM: 2,
        Severity.LOW: 1,
        Severity.INFO: 0,
    }
    decisions = {
        item.finding_id: item for item in state.judge_decisions if item.healer_eligible
    }
    eligible = [item for item in state.confirmed_findings if item.finding_id in decisions]
    if not eligible:
        return None
    finding = max(eligible, key=lambda item: severity_rank[item.severity])
    return finding, decisions[finding.finding_id]


@observe_node(name="healer_agent", actor="healer", as_type="agent")
async def healer_agent(state: QAState) -> dict:
    selected = _selected_healer_pair(state)
    if selected is None:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    finding, decision = selected
    result, proposal, usage_delta = await plan_confirmed_fix(state, finding, decision)
    return {
        "current_phase": PipelinePhase.HEALER,
        "healer_result": result,
        "fix_proposal": proposal,
        "model_usage": _add_usage(state.model_usage, usage_delta),
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.HEALER,
                event_type="HEALER_PLANNED" if proposal else "HEALER_ESCALATED",
                scenario_or_rule_id=finding.finding_id,
                details={
                    "status": result.status.value,
                    "branch": result.branch,
                    "pr_url": result.pr_url,
                    "human_review_required": True,
                    "merge_capability": False,
                },
            ),
        ],
    }


def route_after_healer_agent(state: QAState) -> str:
    if (
        state.fix_proposal is not None
        and state.healer_result is not None
        and state.healer_result.status == HealerStatus.PLANNED
    ):
        return "healer_patch_tool"
    return "human_review_notifier"


@observe_node(name="healer_patch_tool", actor="healer", as_type="tool")
def healer_patch_tool(state: QAState) -> dict:
    result = state.healer_result
    proposal = state.fix_proposal
    if result is None or proposal is None or not result.worktree_path:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    try:
        changed = apply_validated_patch(Path(result.worktree_path), proposal.unified_diff)
        updated = result.model_copy(
            update={
                "status": HealerStatus.PATCHED,
                "investigation_result": (
                    f"Root cause: {proposal.root_cause}. Changed: {', '.join(changed)}"
                ),
            }
        )
        event = "HEALER_PATCH_APPLIED"
    except (HealerError, OSError) as exc:
        updated = result.model_copy(
            update={"status": HealerStatus.ESCALATED, "error": str(exc)}
        )
        event = "HEALER_PATCH_BLOCKED"
    return {
        "current_phase": PipelinePhase.HEALER,
        "healer_result": updated,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.HEALER,
                event_type=event,
                scenario_or_rule_id=result.finding_id,
                details={"status": updated.status.value, "error": updated.error},
            ),
        ],
    }


def route_after_healer_patch(state: QAState) -> str:
    if state.healer_result and state.healer_result.status == HealerStatus.PATCHED:
        return "healer_validation_tool"
    return "human_review_notifier"


@observe_node(name="healer_validation_tool", actor="healer", as_type="tool")
async def healer_validation_tool(state: QAState) -> dict:
    result = state.healer_result
    proposal = state.fix_proposal
    selected = _selected_healer_pair(state)
    if result is None or proposal is None or selected is None or not result.worktree_path:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    finding, _decision = selected
    try:
        commands, outputs = run_validation_profiles(
            Path(result.worktree_path), proposal.validation_profiles
        )
        after_screenshots: list[str] = []
        if finding.kind == FindingKind.UI or "ui" in proposal.validation_profiles:
            after_screenshots, ui_output = await capture_after_ui_evidence(
                state, finding, Path(result.worktree_path)
            )
            commands.append("playwright-mcp: post-fix UI case")
            outputs.append(ui_output)
        updated = result.model_copy(
            update={
                "status": HealerStatus.VALIDATED,
                "validation_commands": commands,
                "validation_output": outputs,
                "after_screenshots": after_screenshots,
            }
        )
        event = "HEALER_VALIDATION_PASSED"
    except (HealerError, OSError) as exc:
        updated = result.model_copy(
            update={"status": HealerStatus.ESCALATED, "error": str(exc)}
        )
        event = "HEALER_VALIDATION_FAILED"
    return {
        "current_phase": PipelinePhase.HEALER,
        "healer_result": updated,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.HEALER,
                event_type=event,
                scenario_or_rule_id=result.finding_id,
                details={
                    "status": updated.status.value,
                    "commands": updated.validation_commands,
                    "after_screenshots": updated.after_screenshots,
                    "error": updated.error,
                },
            ),
        ],
    }


def route_after_healer_validation(state: QAState) -> str:
    if state.healer_result and state.healer_result.status == HealerStatus.VALIDATED:
        return "draft_pr_publisher"
    return "human_review_notifier"


@observe_node(name="draft_pr_publisher", actor="pr_publisher", as_type="tool")
def draft_pr_publisher(state: QAState) -> dict:
    result = state.healer_result
    selected = _selected_healer_pair(state)
    if result is None or selected is None or not result.worktree_path:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    finding, decision = selected
    try:
        updated = publish_draft_pr(
            state, finding, decision, result, Path(result.worktree_path)
        )
        event = "DRAFT_PR_CREATED" if updated.pr_url else "DRAFT_PR_DISABLED"
    except (HealerError, OSError) as exc:
        updated = result.model_copy(
            update={"status": HealerStatus.ESCALATED, "error": str(exc)}
        )
        event = "DRAFT_PR_CREATION_FAILED"
    return {
        "current_phase": PipelinePhase.PR_PUBLISHER,
        "healer_result": updated,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.PR_PUBLISHER,
                event_type=event,
                scenario_or_rule_id=finding.finding_id,
                details={
                    "status": updated.status.value,
                    "pr_url": updated.pr_url,
                    "draft": True,
                    "merge_capability": False,
                    "error": updated.error,
                },
            ),
        ],
    }


@observe_node(name="reviewer_security_tool", actor="reviewer", as_type="tool")
def reviewer_security_tool(state: QAState) -> dict:
    result = state.healer_result
    if result is None or not result.worktree_path:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    try:
        scan = run_reviewer_scans(state, Path(result.worktree_path))
        event = "REVIEWER_GATES_COMPLETED"
        details = {
            "secret_scan_passed": scan.secret_scan_passed,
            "static_scan_passed": scan.static_scan_passed,
            "dependency_scan_passed": scan.dependency_scan_passed,
            "deletion_guard_passed": scan.deletion_guard_passed,
            "findings": scan.findings,
        }
    except (ReviewerError, OSError) as exc:
        from agentic_api_qa.models import SecurityScanReport

        scan = SecurityScanReport(
            secret_scan_passed=False,
            static_scan_passed=False,
            dependency_scan_passed=False,
            deletion_guard_passed=False,
            findings=[f"Reviewer tooling failed closed: {exc}"],
        )
        event = "REVIEWER_GATES_FAILED"
        details = {"error": str(exc)}
    return {
        "current_phase": PipelinePhase.REVIEWER,
        "reviewer_scan": scan,
        "audit_log": [
            *state.audit_log,
            _audit(actor=Actor.REVIEWER, event_type=event, details=details),
        ],
    }


@observe_node(name="reviewer_agent", actor="reviewer", as_type="agent")
async def reviewer_agent(state: QAState) -> dict:
    result = state.healer_result
    scan = state.reviewer_scan
    if result is None or scan is None or not result.worktree_path:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    worktree = Path(result.worktree_path)
    try:
        commit_sha = current_commit(worktree)
        choice = await review_healer_pull_request(
            state, pull_request_diff(state, worktree), scan, commit_sha
        )
        decision = choice.decision
        usage = _add_usage(state.model_usage, choice.usage)
    except (AgentLLMError, ReviewerError, OSError) as exc:
        from agentic_api_qa.models import ReviewerDecision

        decision = ReviewerDecision(
            verdict=ReviewerVerdict.ESCALATE,
            confidence=1.0,
            summary="Automated review stopped fail-closed.",
            security_assessment="Could not complete the required review.",
            deletion_assessment="Could not verify deletion safety.",
            quality_assessment=str(exc),
            required_changes=["Complete a human review before merge."],
        )
        usage = state.model_usage
    review_round = min(
        state.config.limits.reviewer_revision_rounds + 1, state.reviewer_round + 1
    )
    return {
        "current_phase": PipelinePhase.REVIEWER,
        "reviewer_round": review_round,
        "reviewer_decision": decision,
        "model_usage": usage,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.REVIEWER,
                event_type=f"REVIEWER_{decision.verdict.value}",
                details={
                    "round": review_round,
                    "confidence": decision.confidence,
                    "approved_commit_sha": decision.approved_commit_sha,
                    "required_changes": decision.required_changes,
                },
            ),
        ],
    }


def route_after_reviewer(state: QAState) -> str:
    decision = state.reviewer_decision
    if decision is None:
        return "human_review_notifier"
    if decision.verdict == ReviewerVerdict.APPROVE:
        return "reviewer_pr_comment"
    if (
        decision.verdict == ReviewerVerdict.CHANGES_REQUESTED
        and state.reviewer_round <= state.config.limits.reviewer_revision_rounds
    ):
        return "healer_revision_agent"
    return "reviewer_pr_comment"


@observe_node(name="healer_revision_agent", actor="healer", as_type="agent")
async def healer_revision_agent(state: QAState) -> dict:
    selected = _selected_healer_pair(state)
    if selected is None:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    finding, judge_decision = selected
    result, proposal, usage_delta = await plan_reviewer_revision(
        state, finding, judge_decision
    )
    return {
        "current_phase": PipelinePhase.HEALER,
        "healer_result": result,
        "fix_proposal": proposal,
        "model_usage": _add_usage(state.model_usage, usage_delta),
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.HEALER,
                event_type="HEALER_REWORK_PLANNED" if proposal else "HEALER_REWORK_ESCALATED",
                details={"review_round": state.reviewer_round, "status": result.status.value},
            ),
        ],
    }


@observe_node(name="reviewer_pr_comment", actor="reviewer", as_type="tool")
def reviewer_pr_comment(state: QAState) -> dict:
    result = state.healer_result
    decision = state.reviewer_decision
    if result is None or decision is None or not result.worktree_path or not result.pr_url:
        return {"current_phase": PipelinePhase.HUMAN_REVIEW}
    try:
        output = publish_reviewer_comment(state, decision, Path(result.worktree_path))
        event = "REVIEWER_COMMENT_PUBLISHED"
        details = {"verdict": decision.verdict.value, "output": output[:500]}
    except (ReviewerError, OSError) as exc:
        event = "REVIEWER_COMMENT_FAILED"
        details = {"verdict": decision.verdict.value, "error": str(exc)}
    return {
        "current_phase": PipelinePhase.REVIEWER,
        "audit_log": [
            *state.audit_log,
            _audit(actor=Actor.REVIEWER, event_type=event, details=details),
        ],
    }


@observe_node(name="human_review_notifier", actor="human_review", as_type="tool")
async def human_review_notifier(state: QAState) -> dict:
    request = await notify_human_review(state)
    return {
        "current_phase": PipelinePhase.HUMAN_REVIEW,
        "human_review": request,
        "audit_log": [
            *state.audit_log,
            _audit(
                actor=Actor.HUMAN_REVIEW,
                event_type=(
                    "HUMAN_REVIEW_NOTIFIED"
                    if request.slack_message_ts
                    else "HUMAN_REVIEW_RECORDED"
                ),
                details={
                    "required": request.required,
                    "pr_url": request.pr_url,
                    "slack_channel_id": request.slack_channel_id,
                    "notification_error": request.notification_error,
                },
            ),
        ],
    }


@observe_node(name="reporter", actor="reporter", as_type="chain")
def reporter(state: QAState) -> dict:
    finished_at = utc_now()
    happy_failures = sum(item.result == HappyPathResult.FAIL for item in state.happy_path_decisions)
    breaches = sum(item.result == GuardrailResult.BREACHED for item in state.guardrail_decisions)
    inconclusive = sum(item.result == GuardrailResult.INCONCLUSIVE for item in state.guardrail_decisions)
    ui_failures = sum(not item.passed for item in state.ui_results)
    confirmed_bug_count = sum(
        item.verdict == JudgeVerdict.CONFIRMED for item in state.judge_decisions
    )
    if any(error.blocking for error in state.pipeline_errors) or inconclusive:
        overall = OverallResult.ERROR
    elif confirmed_bug_count:
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
        ui_cases_passed=len(state.ui_results) - ui_failures,
        ui_cases_failed=ui_failures,
        ui_mcp_tool_calls=state.ui_mcp_tool_calls,
        ui_screenshots=sum(item.screenshot_captured for item in state.ui_results),
        ui_videos=sum(item.video_captured for item in state.ui_results),
        ui_investigation_cases=max(
            0, len(state.ui_results) - state.config.limits.soft_case_limit
        ),
    )
    summary = (
        f"Happy path: {metrics.happy_path_passed} passed, {metrics.happy_path_failed} failed. "
        f"Guardrails: {metrics.guardrails_held} held, {metrics.guardrails_breached} breached, "
        f"{metrics.guardrails_inconclusive} inconclusive. "
        f"UI smoke: {metrics.ui_cases_passed} passed, {metrics.ui_cases_failed} failed. "
        f"Screenshots: {metrics.ui_screenshots}/{len(state.ui_results)}. "
        f"Failure videos: {metrics.ui_videos}. "
        f"Judge: {confirmed_bug_count} confirmed of {len(state.candidate_findings)} candidates. "
        f"Healer: {state.healer_result.status.value if state.healer_result else 'not requested'}. "
        f"LLM calls: {metrics.llm_calls}."
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
        ui_smoke=state.ui_results,
        design_evaluation=state.design_evaluation,
        candidate_findings=state.candidate_findings,
        judge_decisions=state.judge_decisions,
        healer_result=state.healer_result,
        human_review=state.human_review,
        reviewer_scan=state.reviewer_scan,
        reviewer_decision=state.reviewer_decision,
    )
    publish_report_bundle(report)
    ensure_report_server()
    return {
        "finished_at": finished_at,
        "current_phase": PipelinePhase.COMPLETE,
        "audit_log": audit,
        "report": report,
    }
