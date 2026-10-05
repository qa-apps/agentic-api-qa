"""Evaluation metrics: recall, false positives, time-to-detect, grounding, and cost.

Everything here is a pure function over a finished ``QAState`` and the closed bug
manifest, so the same numbers can be recomputed from a stored report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from agentic_api_qa.models import CandidateFinding, QAState, Severity
from evals.checks import SEVERITY_ORDER
from evals.dataset import SeededBugProfile


class MetricModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeScore(MetricModel):
    name: str
    score: float
    threshold: float
    reason: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.score >= self.threshold


class DecisionOutcome(MetricModel):
    scenario_id: str
    agent: str
    node: str
    action: str
    run_id: str = ""
    case_id: str | None = None
    summary: str = ""
    violations: list[str] = Field(default_factory=list)
    judge_scores: list[JudgeScore] = Field(default_factory=list)
    judge_model: str = ""
    latency_ms: float = 0
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    estimated_cost_usd: float | None = None
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.agent_prompt_tokens
            + self.agent_completion_tokens
            + self.judge_prompt_tokens
            + self.judge_completion_tokens
        )

    @property
    def deepeval_score(self) -> float:
        usable = [item.score for item in self.judge_scores if item.error is None]
        return sum(usable) / len(usable) if usable else 0.0

    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and not self.violations
            and all(item.passed for item in self.judge_scores)
        )


class SeededOutcome(MetricModel):
    profile_id: str
    control: bool = False
    run_id: str = ""
    detection_expectation: str = ""
    detected: bool = False
    judge_confirmed: bool = False
    matched_findings: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)
    reported_findings: int = 0
    actions_to_detect: int | None = None
    max_actions: int = 0
    tool_calls: int = 0
    ui_cases: int = 0
    policy_violations: list[str] = Field(default_factory=list)
    evidence_grounding: float = 0
    severity_accuracy: float | None = None
    judge_scores: list[JudgeScore] = Field(default_factory=list)
    judge_model: str = ""
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    estimated_cost_usd: float | None = None
    latency_ms: float = 0
    langfuse_trace_url: str | None = None
    overall_result: str = ""
    error: str | None = None

    @property
    def deepeval_score(self) -> float:
        usable = [item.score for item in self.judge_scores if item.error is None]
        return sum(usable) / len(usable) if usable else 0.0

    @property
    def total_tokens(self) -> int:
        return (
            self.agent_prompt_tokens
            + self.agent_completion_tokens
            + self.judge_prompt_tokens
            + self.judge_completion_tokens
        )

    @property
    def within_budget(self) -> bool:
        return self.actions_to_detect is None or self.actions_to_detect <= self.max_actions

    @property
    def passed(self) -> bool:
        if self.error is not None or self.policy_violations:
            return False
        if self.control:
            return not self.false_positives
        return (
            self.detected
            and self.within_budget
            and all(score.passed for score in self.judge_scores)
        )


class AggregateMetrics(MetricModel):
    seeded_profiles: int = 0
    bugs_detected: int = 0
    bug_recall: float = 0
    judge_confirmed_recall: float = 0
    false_positive_rate: float = 0
    mean_time_to_detect: float | None = None
    mean_tool_calls: float = 0
    policy_violation_rate: float = 0
    evidence_grounding: float = 0
    severity_accuracy: float | None = None
    decision_scenarios: int = 0
    decision_pass_rate: float = 0
    decision_violation_rate: float = 0
    deepeval_score: float = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    total_latency_ms: float = 0


class EvalRunReport(MetricModel):
    run_label: str
    suite: str
    started_at: datetime
    finished_at: datetime
    judge_model: str
    mutation_seed: int | None = None
    decisions: list[DecisionOutcome] = Field(default_factory=list)
    seeded: list[SeededOutcome] = Field(default_factory=list)
    metrics: AggregateMetrics = Field(default_factory=AggregateMetrics)
    blocking: bool = True

    @property
    def failures(self) -> list[str]:
        return [
            *[item.scenario_id for item in self.decisions if not item.passed],
            *[item.profile_id for item in self.seeded if not item.passed],
        ]

    @property
    def passed(self) -> bool:
        return not self.failures


def detection_expectation(profile: SeededBugProfile) -> str:
    if profile.control:
        return "No functional finding is reported on the clean control copy"
    agent = profile.detection_agent.replace("_", " ").title()
    return f"{agent} detects: {profile.title}"


def finding_text(finding: CandidateFinding) -> str:
    return " ".join(
        [
            finding.title,
            finding.expected,
            finding.actual,
            " ".join(finding.reproduction_steps),
            json.dumps(finding.api_evidence, default=str),
        ]
    ).casefold()


def matches_profile(finding: CandidateFinding, profile: SeededBugProfile) -> bool:
    if finding.design_only:
        return False
    if profile.expected_kinds and finding.kind.value not in profile.expected_kinds:
        return False
    text = finding_text(finding)
    if profile.match_all and not all(word.casefold() in text for word in profile.match_all):
        return False
    return any(word.casefold() in text for word in profile.match_any)


def _actions_to_detect(state: QAState, matched: list[CandidateFinding]) -> int | None:
    api_actions = state.explorer_tool_calls + state.adversary_tool_calls
    ui_order = {result.case_id: index + 1 for index, result in enumerate(state.ui_results)}
    costs: list[int] = []
    for finding in matched:
        iterations = [
            state.evidence[evidence_id].iteration
            for evidence_id in finding.evidence_ids
            if evidence_id in state.evidence
        ]
        if iterations:
            costs.append(min(iterations))
            continue
        case_id = str(finding.api_evidence.get("ui_case_id") or "")
        if case_id in ui_order:
            costs.append(api_actions + ui_order[case_id])
    return min(costs) if costs else None


def _evidence_grounding(state: QAState, findings: list[CandidateFinding]) -> float:
    if not findings:
        return 0.0
    grounded = 0
    for finding in findings:
        resolvable = any(
            evidence_id in state.evidence for evidence_id in finding.evidence_ids
        )
        if resolvable or finding.screenshot_paths:
            grounded += 1
    return grounded / len(findings)


def _severity_accuracy(
    matched: list[CandidateFinding], profile: SeededBugProfile
) -> float | None:
    if not matched:
        return None
    expected = SEVERITY_ORDER.index(profile.expected_severity)
    scores = []
    for finding in matched:
        distance = abs(SEVERITY_ORDER.index(finding.severity) - expected)
        if distance == 0:
            scores.append(1.0)
        elif distance <= profile.severity_tolerance:
            scores.append(0.6)
        else:
            scores.append(0.0)
    return max(scores)


def score_seeded_run(
    *,
    profile: SeededBugProfile,
    state: QAState,
    policy_violations: list[str],
    latency_ms: float,
    langfuse_trace_url: str | None = None,
) -> SeededOutcome:
    """Compare the agent's findings with the closed manifest for one seeded profile."""

    candidates = state.candidate_findings
    functional = [finding for finding in candidates if not finding.design_only]
    matched = [finding for finding in functional if matches_profile(finding, profile)]
    matched_ids = {finding.finding_id for finding in matched}
    confirmed = [
        finding for finding in state.confirmed_findings if finding.finding_id in matched_ids
    ]
    false_positives = [
        f"{finding.kind.value}:{finding.title}"
        for finding in functional
        if finding.finding_id not in matched_ids
    ]
    return SeededOutcome(
        profile_id=profile.profile_id,
        control=profile.control,
        run_id=state.run_id,
        detection_expectation=detection_expectation(profile),
        detected=bool(matched),
        judge_confirmed=bool(confirmed),
        matched_findings=[finding.title for finding in matched],
        false_positives=false_positives,
        reported_findings=len(candidates),
        actions_to_detect=_actions_to_detect(state, matched),
        max_actions=profile.max_actions,
        tool_calls=state.total_tool_calls + state.ui_mcp_tool_calls,
        ui_cases=len(state.ui_results),
        policy_violations=policy_violations,
        evidence_grounding=_evidence_grounding(state, matched or functional),
        severity_accuracy=_severity_accuracy(matched, profile),
        agent_prompt_tokens=state.model_usage.prompt_tokens,
        agent_completion_tokens=state.model_usage.completion_tokens,
        latency_ms=latency_ms,
        langfuse_trace_url=langfuse_trace_url,
        overall_result=(
            state.report.run.overall_result.value if state.report is not None else "ERROR"
        ),
    )


def aggregate(
    decisions: list[DecisionOutcome], seeded: list[SeededOutcome]
) -> AggregateMetrics:
    seeded_bugs = [item for item in seeded if not item.control]
    detected = [item for item in seeded_bugs if item.detected]
    confirmed = [item for item in seeded_bugs if item.judge_confirmed]
    detect_times = [
        item.actions_to_detect for item in detected if item.actions_to_detect is not None
    ]
    reported = sum(item.reported_findings for item in seeded)
    false_positives = sum(len(item.false_positives) for item in seeded)
    grounding = [item.evidence_grounding for item in seeded if item.reported_findings]
    severity = [
        item.severity_accuracy for item in seeded_bugs if item.severity_accuracy is not None
    ]
    judge_scores = [
        *[item.deepeval_score for item in decisions if item.judge_scores],
        *[item.deepeval_score for item in seeded if item.judge_scores],
    ]
    costs = [
        item.estimated_cost_usd
        for item in [*decisions, *seeded]
        if item.estimated_cost_usd is not None
    ]
    return AggregateMetrics(
        seeded_profiles=len(seeded_bugs),
        bugs_detected=len(detected),
        bug_recall=len(detected) / len(seeded_bugs) if seeded_bugs else 0.0,
        judge_confirmed_recall=len(confirmed) / len(seeded_bugs) if seeded_bugs else 0.0,
        false_positive_rate=false_positives / reported if reported else 0.0,
        mean_time_to_detect=sum(detect_times) / len(detect_times) if detect_times else None,
        mean_tool_calls=(
            sum(item.tool_calls for item in seeded) / len(seeded) if seeded else 0.0
        ),
        policy_violation_rate=(
            sum(1 for item in seeded if item.policy_violations) / len(seeded) if seeded else 0.0
        ),
        evidence_grounding=sum(grounding) / len(grounding) if grounding else 0.0,
        severity_accuracy=sum(severity) / len(severity) if severity else None,
        decision_scenarios=len(decisions),
        decision_pass_rate=(
            sum(1 for item in decisions if item.passed) / len(decisions) if decisions else 0.0
        ),
        decision_violation_rate=(
            sum(1 for item in decisions if item.violations) / len(decisions)
            if decisions
            else 0.0
        ),
        deepeval_score=sum(judge_scores) / len(judge_scores) if judge_scores else 0.0,
        total_tokens=(
            sum(item.total_tokens for item in decisions)
            + sum(item.total_tokens for item in seeded)
        ),
        estimated_cost_usd=sum(costs) if costs else None,
        total_latency_ms=(
            sum(item.latency_ms for item in decisions)
            + sum(item.latency_ms for item in seeded)
        ),
    )


def build_report(
    *,
    suite: str,
    run_label: str,
    started_at: datetime,
    judge_model: str,
    decisions: list[DecisionOutcome],
    seeded: list[SeededOutcome],
    mutation_seed: int | None = None,
    blocking: bool = True,
) -> EvalRunReport:
    return EvalRunReport(
        run_label=run_label,
        suite=suite,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        judge_model=judge_model,
        mutation_seed=mutation_seed,
        decisions=decisions,
        seeded=seeded,
        metrics=aggregate(decisions, seeded),
        blocking=blocking,
    )


def severity_label(value: Severity) -> str:
    return value.value
