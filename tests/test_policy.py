from datetime import datetime, timedelta, timezone

from agentic_api_qa.models import Approval, EffectClass, RequestSpec, RunConfig
from agentic_api_qa.policy import evaluate_policy


def request(effect: EffectClass, path: str) -> RequestSpec:
    return RequestSpec(
        operation_id=f"test-{effect.value}",
        method="POST" if effect != EffectClass.READ else "GET",
        path=path,
        effect=effect,
    )


def test_reads_and_allowlisted_ephemeral_calls_are_allowed() -> None:
    config = RunConfig()

    assert evaluate_policy(config, request(EffectClass.READ, "/api/health"), []).allowed
    assert evaluate_policy(config, request(EffectClass.EPHEMERAL, "/api/chat"), []).allowed


def test_unapproved_production_mutation_is_blocked() -> None:
    decision = evaluate_policy(
        RunConfig(), request(EffectClass.MUTATION, "/api/account"), []
    )

    assert not decision.allowed
    assert "approval" in decision.reason.lower()


def test_scoped_unexpired_approval_allows_mutation() -> None:
    approval = Approval(
        scope=["mutation:/api/account"],
        target="/api/account",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    decision = evaluate_policy(
        RunConfig(), request(EffectClass.MUTATION, "/api/account"), [approval]
    )

    assert decision.allowed
    assert decision.approval_id == approval.approval_id
