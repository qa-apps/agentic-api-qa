"""Safety policy evaluated before every HTTP tool call."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from agentic_api_qa.models import Approval, EffectClass, RequestSpec, RunConfig


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    approval_id: str | None = None


def _valid_approval(
    approvals: list[Approval], request: RequestSpec, effect: EffectClass
) -> Approval | None:
    now = datetime.now(timezone.utc)
    wanted = f"{effect.value}:{request.path}"
    for approval in approvals:
        if approval.expires_at <= now or approval.target != request.path:
            continue
        if wanted in approval.scope or effect.value in approval.scope:
            return approval
    return None


def evaluate_policy(
    config: RunConfig, request: RequestSpec, approvals: list[Approval]
) -> PolicyDecision:
    if request.effect == EffectClass.READ:
        return PolicyDecision(allowed=True, reason="Read operation allowed")

    if request.effect == EffectClass.EPHEMERAL:
        if request.path in config.allowed_ephemeral_paths:
            return PolicyDecision(
                allowed=True, reason="Allowlisted nonpersistent operation"
            )
        return PolicyDecision(
            allowed=False, reason="Ephemeral operation is not allowlisted"
        )

    approval = _valid_approval(approvals, request, request.effect)
    if approval is not None:
        return PolicyDecision(
            allowed=True,
            reason="Scoped unexpired approval found",
            approval_id=approval.approval_id,
        )

    if request.effect == EffectClass.MUTATION:
        if not config.production_read_only and not config.require_mutation_approval:
            return PolicyDecision(
                allowed=True, reason="Mutation enabled for controlled environment"
            )
        return PolicyDecision(
            allowed=False, reason="Mutation requires explicit scoped approval"
        )

    return PolicyDecision(
        allowed=False, reason="Destructive operation requires explicit scoped approval"
    )

