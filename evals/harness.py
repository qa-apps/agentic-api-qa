"""Execution harness: run real agents against frozen states and seeded bug copies."""

from __future__ import annotations

import time
from typing import Any

from agentic_api_qa.config import load_run_config
from agentic_api_qa.models import (
    Environment,
    HealerConfig,
    QAState,
    RunConfig,
    SafetyLimits,
)
from agentic_api_qa.observability import langfuse_trace_url
from agentic_api_qa.runner import run_workflow
from evals.checks import decision_violations, run_decision, state_policy_violations
from evals.config import EvalConfig, load_eval_config
from evals.dataset import DecisionScenario, SeededBugProfile
from evals.metrics import (
    DecisionOutcome,
    JudgeScore,
    SeededOutcome,
    detection_expectation,
    score_seeded_run,
)
from evals.reporting import record_eval_observation
from evals.seeded_site import serve_profile


def _record_decision_trace(
    scenario: DecisionScenario,
    state: QAState,
    outcome: DecisionOutcome,
) -> DecisionOutcome:
    record_eval_observation(
        run_id=state.run_id,
        name=f"eval-decision-{scenario.scenario_id}",
        input_data={
            "scenario": scenario.scenario_id,
            "agent": scenario.agent,
            "situation": scenario.situation,
            "evidence": [
                {
                    "id": record.evidence_id,
                    "case": record.scenario_or_rule_id,
                    "method": record.request.method,
                    "url": record.request.url,
                    "status": record.response.status_code,
                    "transport_error": record.response.transport_error,
                    "response_excerpt": record.response.sanitized_body_excerpt[:800],
                }
                for record in state.evidence.values()
            ],
            "previous_tool_calls": state.total_tool_calls + state.ui_mcp_tool_calls,
        },
        output_data=outcome.model_dump(mode="json"),
        metadata={
            "eval_type": "agent_decision",
            "agent": scenario.agent,
            "scenario_id": scenario.scenario_id,
            "passed": outcome.passed,
        },
    )
    return outcome


def _record_seeded_trace(
    profile: SeededBugProfile,
    state: QAState,
    outcome: SeededOutcome,
) -> SeededOutcome:
    record_eval_observation(
        run_id=state.run_id,
        name=f"eval-seeded-{profile.profile_id}",
        input_data={
            "profile_id": profile.profile_id,
            "closed_manifest": profile.description,
            "tool_calls": state.total_tool_calls + state.ui_mcp_tool_calls,
            "evidence": [
                {
                    "id": record.evidence_id,
                    "case": record.scenario_or_rule_id,
                    "method": record.request.method,
                    "url": record.request.url,
                    "status": record.response.status_code,
                    "response_excerpt": record.response.sanitized_body_excerpt[:800],
                }
                for record in state.evidence.values()
            ],
            "ui_results": [item.model_dump(mode="json") for item in state.ui_results],
        },
        output_data=outcome.model_dump(mode="json"),
        metadata={
            "eval_type": "seeded_bug",
            "profile_id": profile.profile_id,
            "passed": outcome.passed,
            "model": state.config.models.explorer_model,
        },
    )
    return outcome


def _estimated_cost(
    *,
    settings: EvalConfig,
    agent_prompt: int,
    agent_completion: int,
    judge_prompt: int,
    judge_completion: int,
) -> float | None:
    rates = (
        settings.agent_input_usd_per_million,
        settings.agent_output_usd_per_million,
        settings.judge_input_usd_per_million,
        settings.judge_output_usd_per_million,
    )
    if any(rate is None for rate in rates):
        return None
    counts = (agent_prompt, agent_completion, judge_prompt, judge_completion)
    return sum(count * float(rate) for count, rate in zip(counts, rates, strict=True)) / 1_000_000


def _judge_scores(verdict: Any) -> list[JudgeScore]:
    return [
        JudgeScore(
            name=item.name,
            score=item.score,
            threshold=item.threshold,
            reason=item.reason,
            error=item.error,
        )
        for item in verdict.scores
    ]


async def run_decision_scenario(
    scenario: DecisionScenario,
    *,
    config: EvalConfig | None = None,
    judge: Any | None = None,
    use_deepeval: bool | None = None,
    run_label: str | None = None,
) -> DecisionOutcome:
    """Run one real agent decision and score it deterministically, then with the judge."""

    settings = config or load_eval_config()
    state = scenario.state()
    if run_label:
        # A fresh run id keeps every nightly run on its own Langfuse trace.
        state = state.model_copy(update={"run_id": f"{state.run_id}-{run_label}"})
    outcome = DecisionOutcome(
        scenario_id=scenario.scenario_id,
        agent=scenario.agent,
        node=scenario.node,
        action="error",
        run_id=state.run_id,
    )
    try:
        decision = await run_decision(scenario, state)
    except Exception as exc:  # noqa: BLE001 - a transport failure is a reportable result
        return outcome.model_copy(update={"error": f"{type(exc).__name__}: {exc}"})

    violations = decision_violations(scenario, state, decision)
    outcome = outcome.model_copy(
        update={
            "action": decision.action,
            "case_id": decision.case_id,
            "summary": decision.summary[:600],
            "violations": violations,
            "latency_ms": decision.latency_ms,
            "agent_prompt_tokens": decision.prompt_tokens,
            "agent_completion_tokens": decision.completion_tokens,
            "estimated_cost_usd": _estimated_cost(
                settings=settings,
                agent_prompt=decision.prompt_tokens,
                agent_completion=decision.completion_tokens,
                judge_prompt=0,
                judge_completion=0,
            ),
        }
    )
    wants_judge = settings.deepeval_enabled if use_deepeval is None else use_deepeval
    if not wants_judge or not scenario.metrics:
        return _record_decision_trace(scenario, state, outcome)
    from evals.judges.decision_quality import judge_decision

    verdict = judge_decision(scenario, state, decision, config=settings, judge=judge)
    outcome = outcome.model_copy(
        update={
            "judge_scores": _judge_scores(verdict),
            "judge_model": verdict.judge_model,
            "judge_prompt_tokens": verdict.judge_prompt_tokens,
            "judge_completion_tokens": verdict.judge_completion_tokens,
            "estimated_cost_usd": _estimated_cost(
                settings=settings,
                agent_prompt=decision.prompt_tokens,
                agent_completion=decision.completion_tokens,
                judge_prompt=verdict.judge_prompt_tokens,
                judge_completion=verdict.judge_completion_tokens,
            ),
        }
    )
    return _record_decision_trace(scenario, state, outcome)


def seeded_run_config(profile: SeededBugProfile, base_url: str, settings: EvalConfig) -> RunConfig:
    """Build a bounded local run configuration for one seeded profile."""

    api_cases = settings.seeded_max_api_cases if profile.surface != "ui" else 3
    ui_cases = settings.seeded_max_ui_cases if profile.surface != "api" else 5
    ui_enabled = settings.seeded_ui_enabled and profile.surface != "api"
    return RunConfig(
        target_name=f"seeded-{profile.profile_id}",
        base_url=base_url,
        environment=Environment.LOCAL,
        production_read_only=True,
        require_mutation_approval=True,
        limits=SafetyLimits(
            explorer_iterations=api_cases,
            adversary_iterations=api_cases,
            explorer_tool_calls=api_cases,
            adversary_tool_calls=api_cases,
            total_tool_calls=api_cases * 2,
            soft_case_limit=max(api_cases, ui_cases),
            hard_case_limit=max(api_cases, ui_cases),
            ui_tool_calls=max(5, ui_cases * 12),
        ),
        models=load_run_config().models,
        healer=HealerConfig(enabled=False),
        ui_enabled=ui_enabled,
        ui_headless=True,
        ui_video="off",
        ui_artifact_dir=f"artifacts/evals/{profile.profile_id}",
    )


async def run_seeded_profile(
    profile: SeededBugProfile,
    *,
    config: EvalConfig | None = None,
    judge: Any | None = None,
    use_deepeval: bool | None = None,
) -> SeededOutcome:
    """Serve the seeded copy, run the full graph against it, and score the findings."""

    settings = config or load_eval_config()
    started = time.perf_counter()
    try:
        with serve_profile(profile, settings) as site:
            run_config = seeded_run_config(profile, site.base_url, settings)
            state = await run_workflow(QAState(config=run_config))
    except Exception as exc:  # noqa: BLE001 - reported as a failed profile, never raised
        return SeededOutcome(
            profile_id=profile.profile_id,
            control=profile.control,
            detection_expectation=detection_expectation(profile),
            max_actions=profile.max_actions,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - started) * 1_000,
        )

    outcome = score_seeded_run(
        profile=profile,
        state=state,
        policy_violations=state_policy_violations(state),
        latency_ms=(time.perf_counter() - started) * 1_000,
        langfuse_trace_url=langfuse_trace_url(state.run_id),
    ).model_copy(
        update={
            "estimated_cost_usd": _estimated_cost(
                settings=settings,
                agent_prompt=state.model_usage.prompt_tokens,
                agent_completion=state.model_usage.completion_tokens,
                judge_prompt=0,
                judge_completion=0,
            )
        }
    )
    wants_judge = settings.deepeval_enabled if use_deepeval is None else use_deepeval
    if not wants_judge or profile.control or not state.candidate_findings:
        return _record_seeded_trace(profile, state, outcome)
    from evals.judges.decision_quality import judge_findings, judge_report

    expectation = (
        f"The reported findings must include this seeded defect: {profile.description} "
        f"Expected severity: {profile.expected_severity.value}. Every claim must be grounded "
        "in the captured request/response or screenshot evidence, and no unrelated defect "
        "may be invented."
    )
    verdict = judge_findings(
        state,
        [finding for finding in state.candidate_findings if not finding.design_only],
        expectation,
        config=settings,
        judge=judge,
    )
    scores = _judge_scores(verdict)
    judge_prompt_tokens = verdict.judge_prompt_tokens
    judge_completion_tokens = verdict.judge_completion_tokens
    if state.report is not None:
        report_verdict = judge_report(
            state.report,
            (
                f"The final report must clearly include the seeded defect: "
                f"{profile.description}; its severity, evidence, coverage, and overall "
                "verdict must be internally consistent."
            ),
            config=settings,
            judge=judge,
        )
        scores.extend(_judge_scores(report_verdict))
        judge_prompt_tokens += report_verdict.judge_prompt_tokens
        judge_completion_tokens += report_verdict.judge_completion_tokens
    outcome = outcome.model_copy(
        update={
            "judge_scores": scores,
            "judge_model": verdict.judge_model,
            "judge_prompt_tokens": judge_prompt_tokens,
            "judge_completion_tokens": judge_completion_tokens,
            "estimated_cost_usd": _estimated_cost(
                settings=settings,
                agent_prompt=state.model_usage.prompt_tokens,
                agent_completion=state.model_usage.completion_tokens,
                judge_prompt=judge_prompt_tokens,
                judge_completion=judge_completion_tokens,
            ),
        }
    )
    return _record_seeded_trace(profile, state, outcome)
