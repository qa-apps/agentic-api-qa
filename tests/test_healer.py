from pathlib import Path

import pytest

from agentic_api_qa.healer import HealerError, apply_validated_patch, heal_confirmed_finding
from agentic_api_qa.human_review import notify_human_review
from agentic_api_qa.llm import JudgeEvaluationChoice, UsageDelta
from agentic_api_qa.models import (
    Actor,
    CandidateFinding,
    FindingKind,
    HealerConfig,
    HealerStatus,
    JudgeDecision,
    JudgeVerdict,
    QAState,
    RunConfig,
    Severity,
    UICaseResult,
)
from agentic_api_qa.nodes import judge, route_after_judge


@pytest.mark.asyncio
async def test_judge_challenges_ui_failure_before_healer(monkeypatch) -> None:
    async def evaluate(_state, findings):
        return JudgeEvaluationChoice(
            decisions=[
                JudgeDecision(
                    finding_id=findings[0].finding_id,
                    verdict=JudgeVerdict.CONFIRMED,
                    confidence=0.93,
                    skeptical_challenge="Ruled out a stale selector by checking the screenshot.",
                    evidence_for=findings[0].screenshot_paths,
                    evidence_against=[],
                    reasoning_summary="The visible control is absent in a reproducible state.",
                    investigation_reason="A functional UI regression has direct visual evidence.",
                    healer_eligible=True,
                )
            ],
            usage=UsageDelta(total_tokens=12),
        )

    monkeypatch.setattr("agentic_api_qa.nodes.evaluate_candidate_findings", evaluate)
    state = QAState(
        config=RunConfig(ui_enabled=True),
        ui_results=[
            UICaseResult(
                case_id="ui-login-modal",
                name="Login modal opens",
                category="authentication",
                passed=False,
                reason="Login dialog did not appear.",
                screenshot_path="/tmp/login-failure.png",
                screenshot_captured=True,
                severity=Severity.HIGH,
            )
        ],
    )
    update = await judge(state)
    judged = state.model_copy(update=update)

    assert judged.judge_decisions[0].skeptical_challenge
    assert judged.judge_decisions[0].healer_eligible is True
    assert judged.confirmed_findings[0].api_evidence["ui_case_id"] == "ui-login-modal"
    assert route_after_judge(judged) == "healer_agent"


def test_patch_tool_blocks_secrets_and_workflow_changes(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SAFE=1\n", encoding="utf-8")
    patch = """diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@ -1 +1 @@
-SAFE=1
+SECRET=2
"""
    with pytest.raises(HealerError, match="protected path"):
        apply_validated_patch(tmp_path, patch)


@pytest.mark.asyncio
async def test_disabled_healer_escalates_without_touching_repo() -> None:
    finding = CandidateFinding(
        source=Actor.EXPLORER,
        kind=FindingKind.API,
        severity=Severity.HIGH,
        title="API contract regression",
        expected="200 JSON",
        actual="500 HTML",
        evidence_ids=["evidence-1"],
    )
    decision = JudgeDecision(
        finding_id=finding.finding_id,
        verdict=JudgeVerdict.CONFIRMED,
        confidence=0.99,
        skeptical_challenge="Retried with deterministic input.",
        reasoning_summary="Product defect confirmed.",
        investigation_reason="Stable server failure.",
        healer_eligible=True,
    )
    state = QAState(config=RunConfig(healer=HealerConfig(enabled=False)))
    result, usage = await heal_confirmed_finding(state, finding, decision)

    assert result.status == HealerStatus.ESCALATED
    assert result.worktree_path is None
    assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_human_review_is_preserved_when_slack_is_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    finding = CandidateFinding(
        source=Actor.UI_EXPLORER,
        kind=FindingKind.UI,
        severity=Severity.HIGH,
        title="Broken login",
        expected="Modal opens",
        actual="No modal",
        screenshot_paths=["/tmp/before.png"],
    )
    state = QAState(
        config=RunConfig(
            healer=HealerConfig(
                enabled=True,
                human_review_channel_id="C0C1L0GNHE0",
            )
        ),
        confirmed_findings=[finding],
    )
    request = await notify_human_review(state)

    assert request.required is True
    assert request.finding_ids == [finding.finding_id]
    assert request.notification_error is not None
