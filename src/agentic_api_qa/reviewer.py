"""Fail-closed pull-request review tools.

The Reviewer can inspect a Healer branch and comment on its draft PR. It cannot
merge, close, or delete a PR, branch, file, or repository resource.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from agentic_api_qa.models import QAState, ReviewerDecision, SecurityScanReport


class ReviewerError(RuntimeError):
    """Raised when a fixed review tool cannot complete safely."""


CRITICAL_FILES = {
    "index.html",
    "script.js",
    "style.css",
    "chat_server.py",
    "rag/main.py",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic credential": re.compile(
        r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{12,}['\"]"
    ),
}
DANGEROUS_PATTERNS = {
    "shell execution": re.compile(r"(?:shell\s*=\s*True|os\.system\s*\()"),
    "dynamic code execution": re.compile(r"\b(?:eval|exec)\s*\("),
    "unsafe DOM injection": re.compile(r"\.innerHTML\s*="),
}


def _run(args: list[str], *, cwd: Path, timeout: int = 180) -> tuple[int, str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).strip()
    return completed.returncode, output[:8_000]


def _base_ref(state: QAState) -> str:
    return f"{state.config.healer.remote}/{state.config.healer.base_branch}"


def pull_request_diff(state: QAState, worktree: Path) -> str:
    code, output = _run(
        ["git", "diff", "--no-ext-diff", "--unified=50", f"{_base_ref(state)}...HEAD"],
        cwd=worktree,
    )
    if code != 0:
        raise ReviewerError(f"Unable to read PR diff: {output}")
    return output


def current_commit(worktree: Path) -> str:
    code, output = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
    if code != 0:
        raise ReviewerError(f"Unable to resolve reviewed commit: {output}")
    return output.strip()


def run_reviewer_scans(state: QAState, worktree: Path) -> SecurityScanReport:
    """Run bounded, deterministic gates against the complete PR diff."""

    base = _base_ref(state)
    commands: list[str] = []
    outputs: list[str] = []
    findings: list[str] = []

    def run(args: list[str], *, timeout: int = 180) -> tuple[int, str]:
        commands.append(" ".join(args))
        code, output = _run(args, cwd=worktree, timeout=timeout)
        outputs.append(output or ("PASS" if code == 0 else f"FAILED ({code})"))
        return code, output

    diff_code, diff = run(
        ["git", "diff", "--no-ext-diff", "--unified=0", f"{base}...HEAD"]
    )
    check_code, check_output = run(["git", "diff", "--check", f"{base}...HEAD"])
    if diff_code != 0:
        findings.append("The pull-request diff could not be read.")
    if check_code != 0:
        findings.append(f"Git whitespace/conflict-marker check failed: {check_output[:500]}")

    _, numstat = run(["git", "diff", "--numstat", f"{base}...HEAD"])
    additions = deletions = 0
    changed_files: list[str] = []
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, path = parts
        changed_files.append(path)
        if added.isdigit():
            additions += int(added)
        if removed.isdigit():
            deletions += int(removed)

    _, name_status = run(["git", "diff", "--name-status", f"{base}...HEAD"])
    deleted_files = [
        line.split("\t", 1)[1]
        for line in name_status.splitlines()
        if line.startswith("D\t") and "\t" in line
    ]
    total = additions + deletions
    deletion_ratio = deletions / total if total else 0.0
    critical_deleted = sorted(set(deleted_files) & CRITICAL_FILES)
    deletion_guard_passed = not critical_deleted and not (
        deletions > 50 and deletion_ratio > 0.35
    )
    if critical_deleted:
        findings.append(f"Critical files deleted: {', '.join(critical_deleted)}")
    if deletions > 50 and deletion_ratio > 0.35:
        findings.append(
            f"Suspicious broad deletion: {deletions} deleted lines ({deletion_ratio:.0%} of diff)."
        )

    added_lines = "\n".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    secret_scan_passed = True
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(added_lines):
            secret_scan_passed = False
            findings.append(f"Potential {label} added to source.")

    static_scan_passed = check_code == 0 and diff_code == 0
    for label, pattern in DANGEROUS_PATTERNS.items():
        if pattern.search(added_lines):
            static_scan_passed = False
            findings.append(f"Security-sensitive pattern added: {label}.")

    python_files = [
        path for path in changed_files if path.endswith(".py") and (worktree / path).is_file()
    ]
    if python_files:
        bandit_code, bandit_output = run(
            [sys.executable, "-m", "bandit", "-q", "-ll", *python_files]
        )
        if bandit_code != 0:
            static_scan_passed = False
            findings.append(f"Bandit reported high-confidence issues: {bandit_output[:800]}")

    dependency_files = [
        path
        for path in changed_files
        if Path(path).name in {"requirements.txt", "requirements-dev.txt"}
        and (worktree / path).is_file()
    ]
    dependency_scan_passed = True
    for dependency_file in dependency_files:
        audit_code, audit_output = run(
            [sys.executable, "-m", "pip_audit", "-r", dependency_file], timeout=300
        )
        if audit_code != 0:
            dependency_scan_passed = False
            findings.append(
                f"Dependency vulnerability scan failed for {dependency_file}: {audit_output[:800]}"
            )

    return SecurityScanReport(
        secret_scan_passed=secret_scan_passed,
        static_scan_passed=static_scan_passed,
        dependency_scan_passed=dependency_scan_passed,
        deletion_guard_passed=deletion_guard_passed,
        additions=additions,
        deletions=deletions,
        deletion_ratio=deletion_ratio,
        deleted_files=deleted_files,
        findings=findings,
        commands=commands,
        outputs=outputs,
    )


def publish_reviewer_comment(
    state: QAState, decision: ReviewerDecision, worktree: Path
) -> str:
    """Publish the agent verdict as a PR comment; never submit/merge/close a review."""

    healer = state.healer_result
    if healer is None or not healer.pr_url:
        raise ReviewerError("Reviewer has no draft PR to comment on")
    body = "\n".join(
        [
            f"## Agentic Reviewer: {decision.verdict.value}",
            "",
            f"**Confidence:** {decision.confidence:.0%}",
            f"**Reviewed commit:** `{decision.approved_commit_sha or current_commit(worktree)}`",
            f"**Security:** {decision.security_assessment}",
            f"**Deletion/scope:** {decision.deletion_assessment}",
            f"**Quality:** {decision.quality_assessment}",
            "",
            decision.summary,
            "",
            "### Required changes",
            *(f"- {item}" for item in decision.required_changes),
            "" if decision.required_changes else "- None",
            "",
            "> This is an automated first-pass review. Human approval is still mandatory; this workflow cannot merge.",
        ]
    )
    code, output = _run(
        ["gh", "pr", "comment", healer.pr_url, "--body", body],
        cwd=worktree,
    )
    if code != 0:
        raise ReviewerError(f"Unable to publish reviewer comment: {output}")
    return output
