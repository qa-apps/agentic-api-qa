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
    JUDGE = "judge"
    REPORTER = "reporter"
    COMPLETE = "complete"
    FAILED = "failed"


class Actor(StrEnum):
    GOVERNOR = "governor"
    EXPLORER = "explorer"
    ADVERSARY = "adversary"
    JUDGE = "judge"
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


class SafetyLimits(QAModel):
    explorer_iterations: int = Field(default=4, ge=1, le=20)
    adversary_iterations: int = Field(default=6, ge=1, le=20)
    explorer_tool_calls: int = Field(default=6, ge=1, le=50)
    adversary_tool_calls: int = Field(default=10, ge=1, le=50)
    total_tool_calls: int = Field(default=16, ge=1, le=100)
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_response_excerpt_bytes: int = Field(default=2_000, ge=100, le=20_000)


class AgentModelConfig(QAModel):
    """Public model settings. The API key is read from the environment, never state."""

    base_url: str = "https://api.z.ai/api/paas/v4/"
    explorer_model: str = "glm-5.3-flash"
    adversary_model: str = "glm-5.3-flash"
    reasoning_effort: Literal["low", "high", "max"] = "high"
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_tokens: int = Field(default=1_200, ge=128, le=16_384)


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
    report: FinalReport | None = None
