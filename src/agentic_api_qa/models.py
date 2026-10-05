"""Pydantic contracts shared by every Agentic API QA graph node."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
HttpStatus = Annotated[int, Field(ge=100, le=599)]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class QAModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class PipelinePhase(StrEnum):
    GOVERNOR = "governor"
    EXPLORER = "explorer"
    ADVERSARY = "adversary"
    UI_EXPLORER = "ui_explorer"
    DESIGN_EVALUATOR = "design_evaluator"
    JUDGE = "judge"
    HEALER = "healer"
    PR_PUBLISHER = "pr_publisher"
    REVIEWER = "reviewer"
    HUMAN_REVIEW = "human_review"
    REPORTER = "reporter"
    COMPLETE = "complete"
    FAILED = "failed"


class Actor(StrEnum):
    GOVERNOR = "governor"
    EXPLORER = "explorer"
    ADVERSARY = "adversary"
    UI_EXPLORER = "ui_explorer"
    DESIGN_EVALUATOR = "design_evaluator"
    JUDGE = "judge"
    HEALER = "healer"
    PR_PUBLISHER = "pr_publisher"
    REVIEWER = "reviewer"
    HUMAN_REVIEW = "human_review"
    REPORTER = "reporter"


class EffectClass(StrEnum):
    READ = "read"
    EPHEMERAL = "ephemeral"
    MUTATION = "mutation"
    DESTRUCTIVE = "destructive"


class HappyPathResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class GuardrailResult(StrEnum):
    HELD = "HELD"
    BREACHED = "BREACHED"
    INCONCLUSIVE = "INCONCLUSIVE"


class OverallResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class Severity(StrEnum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FindingKind(StrEnum):
    API = "api"
    UI = "ui"
    DESIGN = "design"


class JudgeVerdict(StrEnum):
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class HealerStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    PLANNED = "PLANNED"
    BLOCKED = "BLOCKED"
    ESCALATED = "ESCALATED"
    PATCHED = "PATCHED"
    VALIDATED = "VALIDATED"
    DRAFT_PR_CREATED = "DRAFT_PR_CREATED"


class ReviewerVerdict(StrEnum):
    APPROVE = "APPROVE"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    ESCALATE = "ESCALATE"


class SafetyLimits(QAModel):
    explorer_iterations: int = Field(default=30, ge=1, le=30)
    adversary_iterations: int = Field(default=30, ge=1, le=30)
    explorer_tool_calls: int = Field(default=30, ge=1, le=30)
    adversary_tool_calls: int = Field(default=30, ge=1, le=30)
    total_tool_calls: int = Field(default=60, ge=1, le=90)
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_response_excerpt_bytes: int = Field(default=2_000, ge=100, le=20_000)
    soft_case_limit: int = Field(default=20, ge=1, le=30)
    hard_case_limit: int = Field(default=30, ge=1, le=30)
    ui_tool_calls: int = Field(default=300, ge=5, le=500)
    healer_iterations: int = Field(default=8, ge=1, le=12)
    reviewer_revision_rounds: int = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def soft_limit_must_not_exceed_hard_limit(self) -> "SafetyLimits":
        if self.soft_case_limit > self.hard_case_limit:
            raise ValueError("soft_case_limit must be <= hard_case_limit")
        return self


class AgentModelConfig(QAModel):
    """Public model settings. The API key is read from the environment, never state."""

    base_url: str = "https://api.z.ai/api/paas/v4/"
    explorer_model: str = "glm-5.3-flash"
    adversary_model: str = "glm-5.3-flash"
    ui_model: str = "glm-5.3-flash"
    design_model: str = "glm-5.3-flash"
    judge_model: str = "glm-5.3-flash"
    healer_model: str = "glm-5.3-flash"
    reviewer_model: str = "glm-5.3-flash"
    reasoning_effort: Literal["low", "high", "max"] = "high"
    judge_reasoning_effort: Literal["low", "high", "max"] = "max"
    healer_reasoning_effort: Literal["low", "high", "max"] = "high"
    reviewer_reasoning_effort: Literal["low", "high", "max"] = "max"
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_tokens: int = Field(default=1_200, ge=128, le=16_384)


class HealerConfig(QAModel):
    """Fail-closed source-repair settings; secrets never enter graph state."""

    enabled: bool = False
    source_repo: str = "/Users/alex/Projects/alexpavsky"
    remote: str = "origin"
    base_branch: str = "main"
    github_repository: str = "qa-apps/alexpavsky"
    worktree_root: str = ".healer-worktrees"
    create_draft_pr: bool = True
    minimum_judge_confidence: float = Field(default=0.85, ge=0, le=1)
    human_review_channel_id: str | None = None


class RunConfig(QAModel):
    target_name: str = "alexpavsky"
    base_url: HttpUrl = "https://www.alexpavsky.com"
    environment: Environment = Environment.PRODUCTION
    production_read_only: bool = True
    require_mutation_approval: bool = True
    allowed_ephemeral_paths: list[str] = Field(
        default_factory=lambda: ["/api/chat", "/voice-api/api/say"]
    )
    redacted_headers: list[str] = Field(
        default_factory=lambda: [
            "authorization",
            "cookie",
            "proxy-authorization",
            "x-api-key",
        ]
    )
    limits: SafetyLimits = Field(default_factory=SafetyLimits)
    models: AgentModelConfig = Field(default_factory=AgentModelConfig)
    healer: HealerConfig = Field(default_factory=HealerConfig)
    ui_enabled: bool = True
    ui_headless: bool = True
    ui_video: Literal["off", "retain-on-failure", "on"] = "retain-on-failure"
    ui_artifact_dir: str = "artifacts/ui"


class Approval(QAModel):
    approval_id: str = Field(default_factory=lambda: new_id("approval"))
    scope: list[str]
    target: str
    expires_at: datetime


class RequestSpec(QAModel):
    operation_id: str
    method: HttpMethod
    path: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    query: dict[str, str] = Field(default_factory=dict)
    json_body: Any | None = None
    raw_body: str | None = None
    effect: EffectClass = EffectClass.READ

    @model_validator(mode="after")
    def one_body_representation(self) -> "RequestSpec":
        if self.json_body is not None and self.raw_body is not None:
            raise ValueError("Use either json_body or raw_body, not both")
        return self


class PlannedStep(QAModel):
    step_id: str
    name: str
    objective: str
    request: RequestSpec
    expected_status_codes: list[HttpStatus]
    semantic_expectation: str | None = None
    rule_id: str | None = None
    attack_class: str | None = None


class EvidenceRequest(QAModel):
    method: HttpMethod
    url: str
    redacted_headers: dict[str, str] = Field(default_factory=dict)
    body_sha256: str
    sanitized_body_excerpt: str


class EvidenceResponse(QAModel):
    status_code: HttpStatus | None = None
    selected_headers: dict[str, str] = Field(default_factory=dict)
    body_sha256: str = ""
    sanitized_body_excerpt: str = ""
    body_truncated: bool = False
    json_parseable: bool | None = None
    json_type: str | None = None
    json_top_level_keys: list[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0, ge=0)
    transport_error: str | None = None


class EvidenceRecord(QAModel):
    evidence_id: str = Field(default_factory=lambda: new_id("evidence"))
    actor: Actor
    iteration: int = Field(ge=1)
    scenario_or_rule_id: str
    tool_call_id: str = Field(default_factory=lambda: new_id("tool"))
    request: EvidenceRequest
    response: EvidenceResponse
    observed_at: datetime = Field(default_factory=utc_now)
    langsmith_trace_id: str | None = None


class AuditEvent(QAModel):
    event_id: str = Field(default_factory=lambda: new_id("event"))
    timestamp: datetime = Field(default_factory=utc_now)
    actor: Actor
    event_type: str
    iteration: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    scenario_or_rule_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    approval_id: str | None = None
    redacted_details: dict[str, Any] = Field(default_factory=dict)


class AgentReflection(QAModel):
    """Concise, auditable analysis of a tool result (not hidden chain-of-thought)."""

    actor: Actor
    iteration: int = Field(ge=1)
    scenario_or_rule_id: str
    result: str
    interpretation: str
    adaptation: str
    continue_testing: bool
    evidence_id: str


class ModelUsage(QAModel):
    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class PipelineError(QAModel):
    phase: PipelinePhase
    code: str
    message: str
    recoverable: bool = False
    blocking: bool = True


class HappyPathProposal(QAModel):
    step_id: str
    proposed_result: HappyPathResult
    interpretation: str
    evidence_ids: list[str] = Field(min_length=1)


class AdversarialProposal(QAModel):
    rule_id: str
    name: str
    attack_class: str
    proposed_result: GuardrailResult
    severity: Severity
    interpretation: str
    evidence_ids: list[str] = Field(min_length=1)


class HappyPathStepDecision(QAModel):
    step_id: str
    name: str
    result: HappyPathResult
    reason: str
    evidence_ids: list[str] = Field(min_length=1)


class HappyPathReport(QAModel):
    status: HappyPathResult
    steps: list[HappyPathStepDecision]


class GuardrailDecision(QAModel):
    rule_id: str
    name: str
    attack_class: str
    result: GuardrailResult
    severity: Severity
    reason: str
    evidence_ids: list[str] = Field(min_length=1)


class SemanticEvaluation(QAModel):
    case_id: str
    model: str
    passed: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence_ids: list[str]


class UISmokeCase(QAModel):
    case_id: str
    name: str
    category: str
    objective: str
    viewport_width: int = Field(default=1440, ge=320, le=3840)
    viewport_height: int = Field(default=1000, ge=480, le=2160)
    path: str = "/"
    actions: list[dict[str, Any]] = Field(default_factory=list)
    expected_text: list[str] = Field(default_factory=list)
    design_checkpoint: bool = False


class UICaseResult(QAModel):
    case_id: str
    name: str
    category: str
    passed: bool
    reason: str
    screenshot_path: str | None = None
    screenshot_captured: bool = False
    video_path: str | None = None
    video_captured: bool = False
    severity: Severity = Severity.INFO
    investigation_required: bool = False
    snapshot_excerpt: str = ""
    console_errors: list[str] = Field(default_factory=list)
    mcp_tools: list[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0, ge=0)


class DesignScore(QAModel):
    visual_hierarchy: int = Field(ge=1, le=5)
    readability: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    responsive_layout: int = Field(ge=1, le=5)
    accessibility_cues: int = Field(ge=1, le=5)
    interaction_clarity: int = Field(ge=1, le=5)
    summary: str
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    screenshot_paths: list[str] = Field(default_factory=list)


class CandidateFinding(QAModel):
    finding_id: str = Field(default_factory=lambda: new_id("finding"))
    source: Actor
    kind: FindingKind
    severity: Severity
    title: str
    expected: str
    actual: str
    reproduction_steps: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    screenshot_paths: list[str] = Field(default_factory=list)
    api_evidence: dict[str, Any] = Field(default_factory=dict)
    design_only: bool = False


class JudgeDecision(QAModel):
    finding_id: str
    verdict: JudgeVerdict
    confidence: float = Field(ge=0, le=1)
    skeptical_challenge: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    reasoning_summary: str
    investigation_reason: str
    healer_eligible: bool = False


class FixProposal(QAModel):
    finding_id: str
    root_cause: str
    proposed_fix: str
    unified_diff: str
    files_to_change: list[str] = Field(default_factory=list)
    validation_profiles: list[Literal["compile", "api", "ui"]] = Field(
        default_factory=lambda: ["compile"]
    )
    risks: list[str] = Field(default_factory=list)
    rollback: str


class HealerResult(QAModel):
    status: HealerStatus = HealerStatus.NOT_REQUESTED
    finding_id: str | None = None
    investigation_reason: str = ""
    investigation_result: str = ""
    proposed_fix: str = ""
    source_repo: str | None = None
    worktree_path: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    pr_url: str | None = None
    validation_commands: list[str] = Field(default_factory=list)
    validation_output: list[str] = Field(default_factory=list)
    before_screenshots: list[str] = Field(default_factory=list)
    after_screenshots: list[str] = Field(default_factory=list)
    api_evidence: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class HumanReviewRequest(QAModel):
    required: bool = False
    reason: str = ""
    finding_ids: list[str] = Field(default_factory=list)
    pr_url: str | None = None
    slack_channel_id: str | None = None
    slack_message_ts: str | None = None
    evidence_paths: list[str] = Field(default_factory=list)
    notification_error: str | None = None


class SecurityScanReport(QAModel):
    secret_scan_passed: bool
    static_scan_passed: bool
    dependency_scan_passed: bool
    deletion_guard_passed: bool
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    deletion_ratio: float = Field(default=0, ge=0, le=1)
    deleted_files: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class ReviewerDecision(QAModel):
    verdict: ReviewerVerdict
    confidence: float = Field(ge=0, le=1)
    summary: str
    security_assessment: str
    deletion_assessment: str
    quality_assessment: str
    required_changes: list[str] = Field(default_factory=list)
    approved_commit_sha: str | None = None


class ReportRun(QAModel):
    run_id: str
    target: str
    environment: Environment
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0)
    overall_result: OverallResult
    production_read_only: bool


class ReportMetrics(QAModel):
    happy_path_passed: int = Field(ge=0)
    happy_path_failed: int = Field(ge=0)
    guardrails_held: int = Field(ge=0)
    guardrails_breached: int = Field(ge=0)
    guardrails_inconclusive: int = Field(ge=0)
    pipeline_errors: int = Field(ge=0)
    explorer_iterations: int = Field(ge=0)
    adversary_iterations: int = Field(ge=0)
    explorer_tool_calls: int = Field(ge=0)
    adversary_tool_calls: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    llm_prompt_tokens: int = Field(default=0, ge=0)
    llm_completion_tokens: int = Field(default=0, ge=0)
    llm_total_tokens: int = Field(default=0, ge=0)
    ui_cases_passed: int = Field(default=0, ge=0)
    ui_cases_failed: int = Field(default=0, ge=0)
    ui_mcp_tool_calls: int = Field(default=0, ge=0)
    ui_screenshots: int = Field(default=0, ge=0)
    ui_videos: int = Field(default=0, ge=0)
    ui_investigation_cases: int = Field(default=0, ge=0)


class FinalReport(QAModel):
    schema_version: str = "1.0"
    run: ReportRun
    happy_path: HappyPathReport
    adversarial: list[GuardrailDecision]
    summary: str
    metrics: ReportMetrics
    evidence: list[EvidenceRecord]
    audit_log: list[AuditEvent]
    pipeline_errors: list[PipelineError]
    ui_smoke: list[UICaseResult] = Field(default_factory=list)
    design_evaluation: DesignScore | None = None
    candidate_findings: list[CandidateFinding] = Field(default_factory=list)
    judge_decisions: list[JudgeDecision] = Field(default_factory=list)
    healer_result: HealerResult | None = None
    human_review: HumanReviewRequest | None = None
    reviewer_scan: SecurityScanReport | None = None
    reviewer_decision: ReviewerDecision | None = None


class QAState(QAModel):
    """Mutable internal workspace passed through the LangGraph workflow."""

    config: RunConfig = Field(default_factory=RunConfig)
    run_id: str = Field(default_factory=lambda: new_id("run"))
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    current_phase: PipelinePhase = PipelinePhase.GOVERNOR
    approvals: list[Approval] = Field(default_factory=list)

    explorer_plan: list[PlannedStep] = Field(default_factory=list)
    explorer_index: int = Field(default=0, ge=0)
    explorer_iterations: int = Field(default=0, ge=0)
    explorer_tool_calls: int = Field(default=0, ge=0)
    explorer_proposals: list[HappyPathProposal] = Field(default_factory=list)
    explorer_reflections: list[AgentReflection] = Field(default_factory=list)
    explorer_done: bool = False
    explorer_stop_reason: str | None = None

    adversary_plan: list[PlannedStep] = Field(default_factory=list)
    adversary_index: int = Field(default=0, ge=0)
    adversary_iterations: int = Field(default=0, ge=0)
    adversary_tool_calls: int = Field(default=0, ge=0)
    adversary_proposals: list[AdversarialProposal] = Field(default_factory=list)
    adversary_reflections: list[AgentReflection] = Field(default_factory=list)
    adversary_done: bool = False
    adversary_stop_reason: str | None = None

    ui_plan: list[UISmokeCase] = Field(default_factory=list)
    ui_results: list[UICaseResult] = Field(default_factory=list)
    ui_pending_case: UISmokeCase | None = None
    ui_mcp_tool_calls: int = Field(default=0, ge=0)
    ui_plan_summary: str | None = None
    ui_done: bool = False
    ui_stop_reason: str | None = None
    ui_skipped_case_ids: list[str] = Field(default_factory=list)
    ui_serious_bug_case_ids: list[str] = Field(default_factory=list)
    ui_investigation_reasons: list[str] = Field(default_factory=list)
    design_evaluation: DesignScore | None = None

    total_tool_calls: int = Field(default=0, ge=0)
    pending_step: PlannedStep | None = None
    pending_tool_call_id: str | None = None
    pending_decision_summary: str | None = None
    last_evidence_id: str | None = None
    evidence: dict[str, EvidenceRecord] = Field(default_factory=dict)
    audit_log: list[AuditEvent] = Field(default_factory=list)
    pipeline_errors: list[PipelineError] = Field(default_factory=list)
    model_usage: ModelUsage = Field(default_factory=ModelUsage)

    happy_path_decisions: list[HappyPathStepDecision] = Field(default_factory=list)
    guardrail_decisions: list[GuardrailDecision] = Field(default_factory=list)
    semantic_evaluations: list[SemanticEvaluation] = Field(default_factory=list)
    candidate_findings: list[CandidateFinding] = Field(default_factory=list)
    judge_decisions: list[JudgeDecision] = Field(default_factory=list)
    confirmed_findings: list[CandidateFinding] = Field(default_factory=list)
    fix_proposal: FixProposal | None = None
    healer_result: HealerResult | None = None
    human_review: HumanReviewRequest | None = None
    reviewer_round: int = Field(default=0, ge=0, le=3)
    reviewer_scan: SecurityScanReport | None = None
    reviewer_decision: ReviewerDecision | None = None
    report: FinalReport | None = None
