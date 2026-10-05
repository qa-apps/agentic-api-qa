"""DeepEval G-Eval metrics for agent decision and finding quality.

Deterministic mechanics live in ``evals.checks``. These metrics only judge what an assert
cannot: whether the chosen next action is reasonable for the role, whether it is the
highest-value action available, and whether findings are grounded, correctly rated, and
completely reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentic_api_qa.models import CandidateFinding, FinalReport, QAState
from evals.checks import AgentDecision
from evals.config import EvalConfig, load_eval_config
from evals.dataset import DecisionScenario
from evals.judges.zai_judge import EvalJudgeError, build_judge


@dataclass(slots=True)
class MetricScore:
    name: str
    score: float
    threshold: float
    reason: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.score >= self.threshold


@dataclass(slots=True)
class JudgeVerdict:
    scores: list[MetricScore] = field(default_factory=list)
    judge_model: str = ""
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0

    @property
    def judge_tokens(self) -> int:
        return self.judge_prompt_tokens + self.judge_completion_tokens

    @property
    def average(self) -> float:
        usable = [item.score for item in self.scores if item.error is None]
        return sum(usable) / len(usable) if usable else 0.0

    @property
    def failures(self) -> list[MetricScore]:
        return [item for item in self.scores if not item.passed]

    @property
    def errors(self) -> list[str]:
        return [f"{item.name}: {item.error}" for item in self.scores if item.error]


_STEPS: dict[str, tuple[str, list[str]]] = {
    "next_action": (
        "Next-Action Reasonableness",
        [
            "Read the frozen QA state in 'input': which checks already ran, what evidence "
            "exists, which defects are known, and how much budget is left.",
            "Read the agent's actual next decision in 'actual_output' and the reference "
            "expectation in 'expected_output'.",
            "Score high only when the decision is a genuinely useful next step given the "
            "existing evidence: it adds new information, or it correctly stops.",
            "Penalize heavily any decision that repeats covered ground, tests something the "
            "state already proves, or stops while obvious high-value evidence is missing.",
            "Ignore writing style. Judge the action itself, not the confidence of its summary.",
        ],
    ),
    "role_adherence": (
        "Role Adherence",
        [
            "Identify the acting role from 'input': Explorer builds positive coverage, "
            "Adversary designs safe negative guardrail tests, UI Explorer runs UI cases.",
            "Check whether the action in 'actual_output' belongs to that role's mandate.",
            "Penalize an Explorer performing attacks, an Adversary doing plain happy-path "
            "coverage, unsafe destructive or cross-origin intent, and any action that ignores "
            "the stated safety constraints.",
            "Penalize a decision summary that claims evidence the state does not contain.",
        ],
    ),
    "priority": (
        "Action Priority",
        [
            "List the candidate next actions a competent QA engineer would consider in the "
            "state described by 'input'.",
            "Score how close the agent's chosen action in 'actual_output' is to the highest "
            "risk-weighted, highest-information-gain option.",
            "Treat known serious defects as the top priority: scoping a confirmed HIGH or "
            "CRITICAL failure outranks starting unrelated low-risk coverage.",
            "A correct decision to stop scores high when no action carries real information "
            "gain, and low when valuable cheap coverage is still missing.",
        ],
    ),
    "finding_quality": (
        "Finding Quality",
        [
            "Read the observation or finding in 'actual_output' against the evidence in 'input'.",
            "Score high only when the classification is correct, specific, reproducible, and "
            "states expected versus actual behavior.",
            "Penalize vague wording, speculation, restated instructions, duplicated findings, "
            "and any claim the evidence does not support.",
        ],
    ),
    "evidence_grounding": (
        "Evidence Grounding",
        [
            "Extract every factual claim from 'actual_output'.",
            "Verify each claim against the sanitized request/response evidence in 'input'.",
            "Score 1.0 only when every claim is traceable to the cited evidence and the "
            "referenced evidence ids exist.",
            "Penalize invented status codes, invented payload fields, missing evidence links, "
            "and conclusions that go beyond what the evidence shows.",
        ],
    ),
    "severity_accuracy": (
        "Severity Accuracy",
        [
            "Determine the correct severity from the evidence in 'input': data or prompt "
            "disclosure and broken core functionality are HIGH or CRITICAL; cosmetic or "
            "opinion-level issues are LOW or INFO.",
            "Compare that with the severity in 'actual_output' and with 'expected_output'.",
            "Score 1.0 for an exact justified match, about 0.6 for one level off, and near 0 "
            "for inflating a cosmetic issue or downplaying a real security or availability "
            "defect.",
        ],
    ),
    "report_completeness": (
        "Final Report Completeness",
        [
            "Read the final report summary and metrics in 'actual_output'.",
            "Check that it states the overall verdict, the confirmed findings with their "
            "severities, the evidence backing them, and the executed coverage and budget use.",
            "Compare against 'expected_output', which describes what a complete report for "
            "this run must contain.",
            "Penalize missing confirmed defects, unexplained verdicts, and summaries that "
            "contradict the metrics.",
        ],
    ),
}


def available_metrics() -> list[str]:
    return sorted(_STEPS)


def build_metrics(
    names: list[str], *, judge: Any | None = None, threshold: float = 0.6
) -> list[Any]:
    """Construct G-Eval metrics bound to the non-agent judge model."""

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    model = judge or build_judge()
    metrics = []
    for name in names:
        if name not in _STEPS:
            raise EvalJudgeError(f"Unknown decision-quality metric: {name!r}")
        label, steps = _STEPS[name]
        metrics.append(
            GEval(
                name=label,
                evaluation_steps=steps,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                model=model,
                threshold=threshold,
                async_mode=False,
                verbose_mode=False,
            )
        )
    return metrics


def state_digest(state: QAState) -> dict[str, Any]:
    """Compact, judge-visible view of the frozen state. No hidden reasoning is exposed."""

    limits = state.config.limits
    return {
        "target": str(state.config.base_url),
        "environment": state.config.environment.value,
        "production_read_only": state.config.production_read_only,
        "allowed_ephemeral_paths": state.config.allowed_ephemeral_paths,
        "budget": {
            "explorer_iterations_used": state.explorer_iterations,
            "adversary_iterations_used": state.adversary_iterations,
            "ui_cases_executed": len(state.ui_results),
            "total_tool_calls_used": state.total_tool_calls,
            "soft_case_limit": limits.soft_case_limit,
            "hard_case_limit": limits.hard_case_limit,
            "total_tool_calls_limit": limits.total_tool_calls,
        },
        "executed_api_checks": [
            {
                "case": record.scenario_or_rule_id,
                "method": record.request.method,
                "url": record.request.url,
                "status": record.response.status_code,
                "transport_error": record.response.transport_error,
                "response_excerpt": record.response.sanitized_body_excerpt[:600],
            }
            for record in state.evidence.values()
        ],
        "explorer_results": [
            {"case": item.step_id, "result": item.proposed_result.value, "note": item.interpretation}
            for item in state.explorer_proposals
        ],
        "adversary_results": [
            {
                "rule": item.rule_id,
                "result": item.proposed_result.value,
                "severity": item.severity.value,
                "note": item.interpretation,
            }
            for item in state.adversary_proposals
        ],
        "ui_results": [
            {
                "case": item.case_id,
                "passed": item.passed,
                "severity": item.severity.value,
                "reason": item.reason[:300],
            }
            for item in state.ui_results
        ],
        "serious_ui_case_ids": state.ui_serious_bug_case_ids,
        "pending_case": (
            {
                "case": state.pending_step.rule_id or state.pending_step.step_id,
                "method": state.pending_step.request.method,
                "path": state.pending_step.request.path,
                "objective": state.pending_step.objective,
                "expected_status_codes": state.pending_step.expected_status_codes,
            }
            if state.pending_step
            else None
        ),
    }


def _measure(metrics: list[Any], test_case: Any, threshold: float) -> list[MetricScore]:
    scores: list[MetricScore] = []
    for metric in metrics:
        try:
            metric.measure(test_case)
            scores.append(
                MetricScore(
                    name=metric.name,
                    score=float(metric.score or 0.0),
                    threshold=threshold,
                    reason=str(metric.reason or "")[:1_500],
                )
            )
        except Exception as exc:  # noqa: BLE001 - a judge failure must not mask results
            scores.append(
                MetricScore(
                    name=getattr(metric, "name", "unknown"),
                    score=0.0,
                    threshold=threshold,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return scores


def judge_decision(
    scenario: DecisionScenario,
    state: QAState,
    decision: AgentDecision,
    *,
    config: EvalConfig | None = None,
    judge: Any | None = None,
) -> JudgeVerdict:
    """Score one real agent decision with the LLM judge."""

    settings = config or load_eval_config()
    if not scenario.metrics:
        return JudgeVerdict(judge_model=settings.judge_model)
    from deepeval.test_case import LLMTestCase

    threshold = scenario.threshold or settings.decision_threshold
    judge_model = judge or build_judge(settings)
    metrics = build_metrics(scenario.metrics, judge=judge_model, threshold=threshold)
    prompt_tokens_before = getattr(judge_model, "prompt_tokens", 0)
    completion_tokens_before = getattr(judge_model, "completion_tokens", 0)
    test_case = LLMTestCase(
        input=json.dumps(
            {
                "acting_role": scenario.agent,
                "node": scenario.node,
                "situation": scenario.situation,
                "state": state_digest(state),
            },
            default=str,
            indent=2,
        ),
        actual_output=decision.as_output(),
        expected_output=scenario.expected_behavior,
    )
    return JudgeVerdict(
        scores=_measure(metrics, test_case, threshold),
        judge_model=judge_model.get_model_name(),
        judge_prompt_tokens=(
            getattr(judge_model, "prompt_tokens", 0) - prompt_tokens_before
        ),
        judge_completion_tokens=(
            getattr(judge_model, "completion_tokens", 0) - completion_tokens_before
        ),
    )


def judge_findings(
    state: QAState,
    findings: list[CandidateFinding],
    expectation: str,
    *,
    config: EvalConfig | None = None,
    judge: Any | None = None,
    metric_names: list[str] | None = None,
) -> JudgeVerdict:
    """Score the reported findings of a seeded-bug run."""

    settings = config or load_eval_config()
    from deepeval.test_case import LLMTestCase

    threshold = settings.finding_threshold
    judge_model = judge or build_judge(settings)
    metrics = build_metrics(
        metric_names or ["finding_quality", "evidence_grounding", "severity_accuracy"],
        judge=judge_model,
        threshold=threshold,
    )
    prompt_tokens_before = getattr(judge_model, "prompt_tokens", 0)
    completion_tokens_before = getattr(judge_model, "completion_tokens", 0)
    test_case = LLMTestCase(
        input=json.dumps({"state": state_digest(state)}, default=str, indent=2),
        actual_output=json.dumps(
            [
                {
                    "kind": finding.kind.value,
                    "severity": finding.severity.value,
                    "title": finding.title,
                    "expected": finding.expected,
                    "actual": finding.actual,
                    "evidence_ids": finding.evidence_ids,
                    "screenshots": finding.screenshot_paths,
                }
                for finding in findings
            ],
            default=str,
            indent=2,
        ),
        expected_output=expectation,
    )
    return JudgeVerdict(
        scores=_measure(metrics, test_case, threshold),
        judge_model=judge_model.get_model_name(),
        judge_prompt_tokens=(
            getattr(judge_model, "prompt_tokens", 0) - prompt_tokens_before
        ),
        judge_completion_tokens=(
            getattr(judge_model, "completion_tokens", 0) - completion_tokens_before
        ),
    )


def judge_report(
    report: FinalReport,
    expectation: str,
    *,
    config: EvalConfig | None = None,
    judge: Any | None = None,
) -> JudgeVerdict:
    """Score the completeness of the final report for a seeded-bug run."""

    settings = config or load_eval_config()
    from deepeval.test_case import LLMTestCase

    threshold = settings.finding_threshold
    judge_model = judge or build_judge(settings)
    metrics = build_metrics(["report_completeness"], judge=judge_model, threshold=threshold)
    prompt_tokens_before = getattr(judge_model, "prompt_tokens", 0)
    completion_tokens_before = getattr(judge_model, "completion_tokens", 0)
    test_case = LLMTestCase(
        input=json.dumps(
            {
                "overall_result": report.run.overall_result.value,
                "metrics": report.metrics.model_dump(mode="json"),
            },
            default=str,
            indent=2,
        ),
        actual_output=json.dumps(
            {
                "summary": report.summary,
                "confirmed_findings": [
                    {
                        "kind": finding.kind.value,
                        "severity": finding.severity.value,
                        "title": finding.title,
                        "actual": finding.actual,
                    }
                    for finding in report.candidate_findings
                ],
                "judge_decisions": [
                    {
                        "verdict": decision.verdict.value,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning_summary,
                    }
                    for decision in report.judge_decisions
                ],
                "metrics": report.metrics.model_dump(mode="json"),
            },
            default=str,
            indent=2,
        ),
        expected_output=expectation,
    )
    return JudgeVerdict(
        scores=_measure(metrics, test_case, threshold),
        judge_model=judge_model.get_model_name(),
        judge_prompt_tokens=(
            getattr(judge_model, "prompt_tokens", 0) - prompt_tokens_before
        ),
        judge_completion_tokens=(
            getattr(judge_model, "completion_tokens", 0) - completion_tokens_before
        ),
    )
