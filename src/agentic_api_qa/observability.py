"""Langfuse observability for the agent-to-agent QA workflow.

LangSmith remains the graph debugger. Langfuse receives a deterministic trace per
``run_id`` with typed observations for agents, tools, evaluations, and LLM calls.
The helpers are no-ops until both Langfuse keys are configured, so local tests and
LangGraph Studio can still start before credentials exist.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
import os
from typing import Any, Callable, Iterator, Literal, TypeVar, cast

from dotenv import load_dotenv
from langfuse import Langfuse, get_client, propagate_attributes
from opentelemetry import trace as otel_trace

from agentic_api_qa.models import QAState


ObservationType = Literal[
    "span", "agent", "tool", "chain", "retriever", "evaluator", "guardrail",
    "generation", "embedding",
]
F = TypeVar("F", bound=Callable[..., Any])


class _NoopObservation:
    def update(self, **_: Any) -> "_NoopObservation":
        return self


def langfuse_enabled() -> bool:
    """Return whether credentials are present without exposing their values."""

    load_dotenv()
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_trace_id(run_id: str) -> str:
    """Map the external run id to the W3C trace id Langfuse requires."""

    return Langfuse.create_trace_id(seed=run_id)


def _state_input(state: QAState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "phase": state.current_phase.value,
        "target": str(state.config.base_url),
        "pending_api_case": (
            state.pending_step.rule_id or state.pending_step.step_id
            if state.pending_step
            else None
        ),
        "pending_ui_case": state.ui_pending_case.case_id if state.ui_pending_case else None,
        "completed": {
            "explorer": state.explorer_iterations,
            "adversary": state.adversary_iterations,
            "ui": len(state.ui_results),
        },
        "budgets": {
            "soft_case_limit": state.config.limits.soft_case_limit,
            "hard_case_limit": state.config.limits.hard_case_limit,
            "total_tool_calls": state.config.limits.total_tool_calls,
            "ui_tool_calls": state.config.limits.ui_tool_calls,
        },
    }


def _state_metadata(state: QAState, actor: str, node_name: str) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "actor": actor,
        "node": node_name,
        "phase_before": state.current_phase.value,
        "target_name": state.config.target_name,
        "production_read_only": state.config.production_read_only,
        "explorer_tool_calls": state.explorer_tool_calls,
        "adversary_tool_calls": state.adversary_tool_calls,
        "ui_mcp_tool_calls": state.ui_mcp_tool_calls,
        "llm_calls_before": state.model_usage.calls,
        "candidate_findings": len(state.candidate_findings),
        "confirmed_findings": len(state.confirmed_findings),
        "healer_status": state.healer_result.status.value if state.healer_result else None,
    }


def _node_output(node_name: str, state: QAState, result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"node": node_name, "result_type": type(result).__name__}
    last_event = None
    audit = result.get("audit_log")
    if audit:
        event = audit[-1]
        last_event = {
            "actor": event.actor.value,
            "event_type": event.event_type,
            "case_id": event.scenario_or_rule_id,
            "details": event.redacted_details,
        }
    return {
        "node": node_name,
        "phase_after": getattr(result.get("current_phase"), "value", None),
        "handoff": _handoff(node_name, state, result),
        "last_event": last_event,
        "state_updates": sorted(result.keys()),
    }


def _handoff(node_name: str, state: QAState, result: dict[str, Any]) -> str | None:
    if node_name == "safety_governor":
        return "explorer_agent"
    if node_name == "explorer_agent":
        return "adversary_agent" if result.get("explorer_done") else "explorer_http_tool"
    if node_name == "explorer_http_tool":
        return "explorer_observe"
    if node_name == "explorer_observe":
        return "adversary_agent" if result.get("explorer_done") else "explorer_agent"
    if node_name == "adversary_agent":
        return "ui_explorer_agent" if result.get("adversary_done") else "adversary_http_tool"
    if node_name == "adversary_http_tool":
        return "adversary_observe"
    if node_name == "adversary_observe":
        return "ui_explorer_agent" if result.get("adversary_done") else "adversary_agent"
    if node_name == "ui_explorer_agent":
        return "design_evaluator" if result.get("ui_done") else "playwright_mcp_tool"
    if node_name == "playwright_mcp_tool":
        return "ui_observe"
    if node_name == "ui_observe":
        return "ui_explorer_agent"
    if node_name == "design_evaluator":
        return "judge"
    if node_name == "judge":
        confirmed = result.get("confirmed_findings") or []
        if not confirmed:
            return "json_reporter"
        decisions = result.get("judge_decisions") or []
        return (
            "healer_agent"
            if any(getattr(item, "healer_eligible", False) for item in decisions)
            else "human_review_notifier"
        )
    if node_name == "healer_agent":
        return "healer_patch_tool" if result.get("fix_proposal") else "human_review_notifier"
    if node_name == "healer_patch_tool":
        healer = result.get("healer_result")
        return (
            "healer_validation_tool"
            if healer is not None and healer.status.value == "PATCHED"
            else "human_review_notifier"
        )
    if node_name == "healer_validation_tool":
        healer = result.get("healer_result")
        return (
            "draft_pr_publisher"
            if healer is not None and healer.status.value == "VALIDATED"
            else "human_review_notifier"
        )
    if node_name == "draft_pr_publisher":
        return "reviewer_security_tool"
    if node_name == "reviewer_security_tool":
        return "reviewer_agent"
    if node_name == "reviewer_agent":
        decision = result.get("reviewer_decision")
        if decision is not None and decision.verdict.value == "CHANGES_REQUESTED":
            return "healer_revision_agent"
        return "reviewer_pr_comment"
    if node_name == "healer_revision_agent":
        return "healer_patch_tool" if result.get("fix_proposal") else "human_review_notifier"
    if node_name == "reviewer_pr_comment":
        return "human_review_notifier"
    if node_name == "human_review_notifier":
        return "json_reporter"
    if node_name == "reporter":
        return "complete"
    return None


@contextmanager
def node_observation(
    state: QAState,
    *,
    name: str,
    actor: str,
    as_type: ObservationType,
) -> Iterator[Any]:
    """Create a typed node observation attached to this run's trace."""

    if not langfuse_enabled():
        yield _NoopObservation()
        return
    client = get_client()
    current_context = otel_trace.get_current_span().get_span_context()
    trace_context = (
        None
        if current_context.is_valid
        else {"trace_id": langfuse_trace_id(state.run_id)}
    )
    with client.start_as_current_observation(
        trace_context=trace_context,
        name=name,
        as_type=as_type,
        input=_state_input(state),
        metadata=_state_metadata(state, actor, name),
    ) as observation:
        with propagate_attributes(
            session_id=state.run_id,
            tags=["agentic-qa", state.config.target_name, actor],
            trace_name="agentic-qa-workflow",
            environment=state.config.environment.value,
            metadata={
                "run_id": state.run_id,
                "target": state.config.target_name,
                "workflow_version": "0.1.0",
            },
        ):
            try:
                yield observation
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=f"{type(exc).__name__}: {exc}",
                )
                raise


def observe_node(
    *, name: str, actor: str, as_type: ObservationType = "span"
) -> Callable[[F], F]:
    """Trace a LangGraph node and record its explicit state handoff."""

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(state: QAState, *args: Any, **kwargs: Any) -> Any:
                with node_observation(state, name=name, actor=actor, as_type=as_type) as obs:
                    result = await func(state, *args, **kwargs)
                    obs.update(output=_node_output(name, state, result))
                    return result

            return cast(F, async_wrapper)

        @wraps(func)
        def sync_wrapper(state: QAState, *args: Any, **kwargs: Any) -> Any:
            with node_observation(state, name=name, actor=actor, as_type=as_type) as obs:
                result = func(state, *args, **kwargs)
                obs.update(output=_node_output(name, state, result))
                return result

        return cast(F, sync_wrapper)

    return decorator


def _message_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            safe_content: Any = content[:2_000]
        elif isinstance(content, list):
            safe_content = [
                {"type": item.get("type"), "has_image": item.get("type") == "image_url"}
                if isinstance(item, dict)
                else {"type": type(item).__name__}
                for item in content
            ]
        else:
            safe_content = None
        summary.append({"role": message.get("role"), "content": safe_content})
    return summary


@contextmanager
def generation_observation(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    reasoning_effort: str,
) -> Iterator[Any]:
    """Trace one real provider call while excluding secrets and hidden reasoning."""

    if not langfuse_enabled():
        yield _NoopObservation()
        return
    tool_names = [
        tool.get("function", {}).get("name")
        for tool in (tools or [])
        if isinstance(tool, dict)
    ]
    client = get_client()
    with client.start_as_current_observation(
        name="zai_chat_completion",
        as_type="generation",
        model=model,
        input={"messages": _message_summary(messages), "tools": tool_names},
        model_parameters={"reasoning_effort": reasoning_effort},
        metadata={"provider": "z.ai", "hidden_reasoning_recorded": False},
    ) as observation:
        try:
            yield observation
        except Exception as exc:
            observation.update(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
            )
            raise


@contextmanager
def workflow_observation(state: QAState) -> Iterator[Any]:
    """Create the root trace for CLI runs and propagate run-level attributes."""

    if not langfuse_enabled():
        yield _NoopObservation()
        return
    client = get_client()
    with client.start_as_current_observation(
        trace_context={"trace_id": langfuse_trace_id(state.run_id)},
        name="agentic-qa-workflow",
        as_type="chain",
        input=_state_input(state),
        metadata={
            "run_id": state.run_id,
            "soft_case_limit": state.config.limits.soft_case_limit,
            "hard_case_limit": state.config.limits.hard_case_limit,
            "production_read_only": state.config.production_read_only,
        },
    ) as observation:
        with propagate_attributes(
            session_id=state.run_id,
            tags=["agentic-qa", state.config.target_name, "full-workflow"],
            trace_name="agentic-qa-workflow",
            environment=state.config.environment.value,
            metadata={"run_id": state.run_id, "target": state.config.target_name},
        ):
            try:
                yield observation
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=f"{type(exc).__name__}: {exc}",
                )
                raise


def score_completed_run(state: QAState) -> None:
    """Publish deterministic QA metrics as Langfuse trace scores."""

    if not langfuse_enabled() or state.report is None:
        return
    client = get_client()
    trace_id = langfuse_trace_id(state.run_id)
    metrics = state.report.metrics
    ui_total = metrics.ui_cases_passed + metrics.ui_cases_failed
    guardrail_total = metrics.guardrails_held + metrics.guardrails_breached + metrics.guardrails_inconclusive
    scores: list[tuple[str, float, str]] = [
        ("workflow_pass", float(state.report.run.overall_result.value == "PASS"), "BOOLEAN"),
        ("ui_pass_rate", metrics.ui_cases_passed / ui_total if ui_total else 0.0, "NUMERIC"),
        ("screenshot_coverage", metrics.ui_screenshots / ui_total if ui_total else 0.0, "NUMERIC"),
        ("guardrail_hold_rate", metrics.guardrails_held / guardrail_total if guardrail_total else 0.0, "NUMERIC"),
    ]
    if state.design_evaluation is not None:
        design = state.design_evaluation
        average = sum(
            [
                design.visual_hierarchy,
                design.readability,
                design.consistency,
                design.responsive_layout,
                design.accessibility_cues,
                design.interaction_clarity,
            ]
        ) / 30.0
        scores.append(("design_quality", average, "NUMERIC"))
    for name, value, data_type in scores:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=cast(Any, data_type),
            environment=state.config.environment.value,
            metadata={"run_id": state.run_id},
        )


def flush_langfuse() -> None:
    """Flush queued observations for short-lived CLI processes."""

    if langfuse_enabled():
        get_client().flush()


def langfuse_trace_url(run_id: str) -> str | None:
    if not langfuse_enabled():
        return None
    return get_client().get_trace_url(trace_id=langfuse_trace_id(run_id))
