import json
import subprocess
from pathlib import Path

import pytest

from agentic_api_qa import llm
from agentic_api_qa.llm import review_healer_pull_request
from agentic_api_qa.models import (
    QAState,
    ReviewerDecision,
    ReviewerVerdict,
    SecurityScanReport,
)
from agentic_api_qa.nodes import route_after_reviewer
from agentic_api_qa.reviewer import run_reviewer_scans


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _review_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "app"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "index.html").write_text("safe\n", encoding="utf-8")
    (repo / "notes.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    _git(repo, "switch", "-c", "agent-fix/test")
    return repo


def test_reviewer_secret_gate_blocks_added_credential(tmp_path: Path) -> None:
    repo = _review_repo(tmp_path)
    (repo / "notes.txt").write_text(
        'api_key = "abcdefghijklmnop-secret"\n', encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "unsafe")

    scan = run_reviewer_scans(QAState(), repo)

    assert scan.secret_scan_passed is False
    assert any("credential" in item for item in scan.findings)


def test_reviewer_deletion_guard_blocks_critical_file(tmp_path: Path) -> None:
    repo = _review_repo(tmp_path)
    (repo / "index.html").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "delete critical page")

    scan = run_reviewer_scans(QAState(), repo)

    assert scan.deletion_guard_passed is False
    assert scan.deleted_files == ["index.html"]


@pytest.mark.asyncio
async def test_llm_cannot_approve_a_failed_deterministic_gate(monkeypatch) -> None:
    async def fake_completion(**_kwargs):
        return {
            "message": {
                "content": json.dumps(
                    {
                        "verdict": "APPROVE",
                        "confidence": 0.99,
                        "summary": "looks good",
                        "security_assessment": "safe",
                        "deletion_assessment": "safe",
                        "quality_assessment": "good",
                        "required_changes": [],
                        "approved_commit_sha": "wrong",
                    }
                )
            },
            "usage": {},
        }

    monkeypatch.setattr(llm, "_chat_completion", fake_completion)
    scan = SecurityScanReport(
        secret_scan_passed=False,
        static_scan_passed=True,
        dependency_scan_passed=True,
        deletion_guard_passed=True,
        findings=["secret"],
    )

    result = await review_healer_pull_request(QAState(), "diff", scan, "abc123")

    assert result.decision.verdict == ReviewerVerdict.CHANGES_REQUESTED
    assert result.decision.approved_commit_sha is None


def test_reviewer_routes_rework_then_human_handoff() -> None:
    decision = ReviewerDecision(
        verdict=ReviewerVerdict.CHANGES_REQUESTED,
        confidence=0.9,
        summary="changes needed",
        security_assessment="pass",
        deletion_assessment="pass",
        quality_assessment="needs a regression test",
        required_changes=["add test"],
    )
    state = QAState(reviewer_round=1, reviewer_decision=decision)
    assert route_after_reviewer(state) == "healer_revision_agent"

    exhausted = state.model_copy(update={"reviewer_round": 3})
    assert route_after_reviewer(exhausted) == "reviewer_pr_comment"

    approved = state.model_copy(
        update={
            "reviewer_decision": decision.model_copy(
                update={"verdict": ReviewerVerdict.APPROVE}
            )
        }
    )
    assert route_after_reviewer(approved) == "reviewer_pr_comment"
