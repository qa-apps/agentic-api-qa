"""Bounded source-repair tools for Judge-confirmed findings.

The Healer never edits the user's checkout. It starts from the configured remote
base in an isolated worktree, applies one validated patch, and may open a *draft*
pull request. There is deliberately no merge capability in this module.
"""

from __future__ import annotations

import asyncio
import re
import socket
import subprocess
import sys
from pathlib import Path

from agentic_api_qa.llm import (
    AgentLLMError,
    UsageDelta,
    propose_healer_fix,
    propose_healer_revision,
)
from agentic_api_qa.mcp_browser import execute_ui_smoke_case
from agentic_api_qa.models import (
    CandidateFinding,
    FindingKind,
    FixProposal,
    HealerResult,
    HealerStatus,
    JudgeDecision,
    QAState,
)
from agentic_api_qa.ui_catalog import alexpavsky_ui_smoke_catalog


class HealerError(RuntimeError):
    """A fail-closed Healer safety or execution error."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PARTS = {".git", ".github", ".env", "AGENTS.md"}
SOURCE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".html", ".css", ".json", ".md"}


def _run(
    args: list[str], *, cwd: Path, input_text: str | None = None, timeout: int = 120
) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        raise HealerError(f"{' '.join(args[:3])} failed ({completed.returncode}): {output[:3000]}")
    return output


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")[:48] or "finding"


def _worktree_root(state: QAState) -> Path:
    configured = Path(state.config.healer.worktree_root)
    root = configured if configured.is_absolute() else PROJECT_ROOT / configured
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def prepare_isolated_worktree(state: QAState, finding: CandidateFinding) -> tuple[Path, str]:
    config = state.config.healer
    source_repo = Path(config.source_repo).resolve()
    if not (source_repo / ".git").exists():
        raise HealerError(f"Configured source repository is not a Git checkout: {source_repo}")
    branch = f"agent-fix/{_safe_slug(state.run_id)}-{_safe_slug(finding.finding_id)[-12:]}"
    destination = (_worktree_root(state) / _safe_slug(branch.replace("/", "-"))).resolve()
    if destination.exists():
        raise HealerError(f"Isolated worktree already exists; refusing to overwrite: {destination}")
    _run(["git", "fetch", config.remote, config.base_branch], cwd=source_repo)
    remote_base = f"{config.remote}/{config.base_branch}"
    _run(
        ["git", "worktree", "add", "-b", branch, str(destination), remote_base],
        cwd=source_repo,
    )
    return destination, branch


def build_code_context(worktree: Path, finding: CandidateFinding, limit: int = 36_000) -> str:
    """Return bounded source context, favoring files that describe the affected surface."""

    manifest = _run(["git", "ls-files"], cwd=worktree).splitlines()
    candidates = [
        path
        for path in manifest
        if Path(path).suffix.lower() in SOURCE_SUFFIXES
        and not any(part in PROTECTED_PARTS for part in Path(path).parts)
    ]
    priority_names = {
        "README.md",
        "index.html",
        "script.js",
        "style.css",
        "chat_server.py",
        "llm.py",
        "rag/main.py",
    }
    keywords = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", f"{finding.title} {finding.actual}")
    }

    def score(path: str) -> tuple[int, int, str]:
        folded = path.casefold()
        keyword_hits = sum(token in folded for token in keywords)
        return (int(path in priority_names), keyword_hits, path)

    chunks: list[str] = []
    used = 0
    for relative in sorted(candidates, key=score, reverse=True):
        path = (worktree / relative).resolve()
        if worktree not in path.parents or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunk = f"\n--- {relative} ---\n{text[:12_000]}"
        if used + len(chunk) > limit:
            continue
        chunks.append(chunk)
        used += len(chunk)
        if used >= limit - 2_000:
            break
    return "".join(chunks)


def _diff_paths(unified_diff: str) -> list[str]:
    paths: list[str] = []
    for line in unified_diff.splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].split("\t", 1)[0]
        if raw == "/dev/null":
            continue
        relative = raw[2:] if raw.startswith("b/") else raw
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise HealerError(f"Patch path escapes the worktree: {relative}")
        if any(part in PROTECTED_PARTS or part.startswith(".env") for part in path.parts):
            raise HealerError(f"Patch targets a protected path: {relative}")
        paths.append(relative)
    if not paths:
        raise HealerError("Healer did not produce a file-changing unified diff")
    if len(set(paths)) > 8:
        raise HealerError("Healer patch exceeds the eight-file safety limit")
    return sorted(set(paths))


def apply_validated_patch(worktree: Path, unified_diff: str) -> list[str]:
    if len(unified_diff.encode("utf-8")) > 50_000:
        raise HealerError("Healer patch exceeds the 50 KB safety limit")
    paths = _diff_paths(unified_diff)
    _run(["git", "apply", "--check", "-"], cwd=worktree, input_text=unified_diff)
    _run(["git", "apply", "-"], cwd=worktree, input_text=unified_diff)
    _run(["git", "diff", "--check"], cwd=worktree)
    return paths


def run_validation_profiles(worktree: Path, profiles: list[str]) -> tuple[list[str], list[str]]:
    """Run only named, fixed command profiles; the LLM cannot provide shell commands."""

    commands: list[list[str]] = []
    if "compile" in profiles:
        python_files = [path for path in ("chat_server.py", "llm.py", "rag/main.py") if (worktree / path).exists()]
        if python_files:
            commands.append([sys.executable, "-m", "py_compile", *python_files])
        commands.append(["git", "diff", "--check"])
    if "api" in profiles:
        commands.append([sys.executable, "-m", "py_compile", "chat_server.py"])
    if "ui" in profiles:
        commands.append(["git", "diff", "--check"])
    rendered: list[str] = []
    outputs: list[str] = []
    for command in commands:
        if command[-1] == "chat_server.py" and not (worktree / "chat_server.py").exists():
            continue
        rendered.append(" ".join(command))
        outputs.append(_run(command, cwd=worktree, timeout=180)[:4_000] or "PASS")
    return rendered, outputs


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def capture_after_ui_evidence(
    state: QAState, finding: CandidateFinding, worktree: Path
) -> tuple[list[str], str]:
    case_id = str(finding.api_evidence.get("ui_case_id") or "ui-home-desktop")
    catalog = {case.case_id: case for case in alexpavsky_ui_smoke_catalog()}
    case = catalog.get(case_id, catalog["ui-home-desktop"])
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=worktree,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await asyncio.sleep(0.4)
        local_config = state.config.model_copy(
            update={
                "base_url": f"http://127.0.0.1:{port}",
                "ui_artifact_dir": str(PROJECT_ROOT / "artifacts" / "healer-after"),
            }
        )
        result, _ = await execute_ui_smoke_case(
            run_config=local_config,
            run_id=state.run_id,
            case=case,
        )
        screenshots = [result.screenshot_path] if result.screenshot_path else []
        if not result.passed:
            raise HealerError(f"Post-fix UI validation failed: {result.reason}")
        return screenshots, result.reason
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()


def publish_draft_pr(
    state: QAState,
    finding: CandidateFinding,
    decision: JudgeDecision,
    result: HealerResult,
    worktree: Path,
) -> HealerResult:
    """Commit, push, and create a draft PR. This module intentionally cannot merge."""

    config = state.config.healer
    if not config.create_draft_pr:
        return result.model_copy(update={"status": HealerStatus.VALIDATED})
    status = _run(["git", "status", "--porcelain"], cwd=worktree)
    if not status.strip():
        raise HealerError("Validation passed but the Healer produced no source changes")
    _run(["git", "add", "--all"], cwd=worktree)
    _run(
        [
            "git",
            "-c",
            "user.name=AP Agentic QA Healer",
            "-c",
            "user.email=qa-apps@users.noreply.github.com",
            "commit",
            "-m",
            f"fix: {finding.title[:72]}",
        ],
        cwd=worktree,
    )
    commit_sha = _run(["git", "rev-parse", "HEAD"], cwd=worktree).strip()
    _run(["git", "push", "--set-upstream", config.remote, result.branch or ""], cwd=worktree, timeout=180)
    body = f"""## Judge-confirmed finding

**Finding:** {finding.title}
**Severity:** {finding.severity.value}
**Judge confidence:** {decision.confidence:.2f}
**Skeptical challenge:** {decision.skeptical_challenge}

## Investigation

{result.investigation_reason}

{result.investigation_result}

## Proposed fix

{result.proposed_fix}

## Validation

{chr(10).join(f'- `{command}` — PASS' for command in result.validation_commands)}

## Evidence

- Before screenshots: {', '.join(result.before_screenshots) or 'not applicable'}
- After screenshots: {', '.join(result.after_screenshots) or 'not applicable'}
- API evidence: {result.api_evidence or 'not applicable'}

> Draft PR created by the AP Agentic QA Healer. **Human review is required.**
> This workflow has no merge capability.
"""
    if result.pr_url:
        pr_url = result.pr_url
        _run(
            ["gh", "pr", "edit", pr_url, "--body", body],
            cwd=worktree,
            timeout=180,
        )
    else:
        pr_url = _run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                config.github_repository,
                "--draft",
                "--base",
                config.base_branch,
                "--head",
                result.branch or "",
                "--title",
                f"fix: {finding.title[:72]}",
                "--body",
                body,
            ],
            cwd=worktree,
            timeout=180,
        ).strip()
    return result.model_copy(
        update={
            "status": HealerStatus.DRAFT_PR_CREATED,
            "commit_sha": commit_sha,
            "pr_url": pr_url.splitlines()[-1] if pr_url else None,
        }
    )


async def plan_reviewer_revision(
    state: QAState, finding: CandidateFinding, decision: JudgeDecision
) -> tuple[HealerResult, FixProposal | None, UsageDelta]:
    """Ask the Healer for one bounded incremental response to Reviewer feedback."""

    result = state.healer_result
    reviewer = state.reviewer_decision
    if result is None or reviewer is None or not result.worktree_path:
        raise HealerError("Reviewer revision is missing the existing worktree or decision")
    worktree = Path(result.worktree_path)
    try:
        current_diff = _run(
            [
                "git",
                "diff",
                "--unified=30",
                f"{state.config.healer.remote}/{state.config.healer.base_branch}...HEAD",
            ],
            cwd=worktree,
        )
        context = (build_code_context(worktree, finding, limit=24_000) + "\n--- CURRENT PR DIFF ---\n" + current_diff)[-50_000:]
        choice = await propose_healer_revision(
            state, finding, decision, reviewer, context
        )
        if not choice.proposal.unified_diff.strip():
            raise HealerError(
                choice.proposal.proposed_fix or "Healer could not safely satisfy review"
            )
        return (
            result.model_copy(
                update={
                    "status": HealerStatus.PLANNED,
                    "investigation_result": (
                        f"Reviewer revision {state.reviewer_round}: "
                        f"{choice.proposal.root_cause}"
                    ),
                    "proposed_fix": choice.proposal.proposed_fix,
                    "error": None,
                }
            ),
            choice.proposal,
            choice.usage,
        )
    except (AgentLLMError, HealerError, OSError, subprocess.SubprocessError) as exc:
        return (
            result.model_copy(
                update={
                    "status": HealerStatus.ESCALATED,
                    "error": f"Reviewer rework stopped fail-closed: {exc}",
                }
            ),
            None,
            UsageDelta(),
        )


async def plan_confirmed_fix(
    state: QAState, finding: CandidateFinding, decision: JudgeDecision
) -> tuple[HealerResult, FixProposal | None, UsageDelta]:
    """Prepare an isolated worktree and ask the Healer for one bounded patch plan."""

    if not state.config.healer.enabled:
        return (
            HealerResult(
                status=HealerStatus.ESCALATED,
                finding_id=finding.finding_id,
                investigation_reason=decision.investigation_reason,
                error="Healer is disabled by configuration.",
                before_screenshots=finding.screenshot_paths,
                api_evidence=finding.api_evidence,
            ),
            None,
            UsageDelta(),
        )
    try:
        worktree, branch = prepare_isolated_worktree(state, finding)
        context = build_code_context(worktree, finding)
        choice = await propose_healer_fix(state, finding, decision, context)
        if not choice.proposal.unified_diff.strip():
            raise HealerError(
                choice.proposal.proposed_fix or "Healer lacked sufficient code context"
            )
        return (
            HealerResult(
                status=HealerStatus.PLANNED,
                finding_id=finding.finding_id,
                investigation_reason=decision.investigation_reason,
                investigation_result=f"Proposed root cause: {choice.proposal.root_cause}",
                proposed_fix=choice.proposal.proposed_fix,
                source_repo=state.config.healer.source_repo,
                worktree_path=str(worktree),
                branch=branch,
                before_screenshots=finding.screenshot_paths,
                api_evidence=finding.api_evidence,
            ),
            choice.proposal,
            choice.usage,
        )
    except (AgentLLMError, HealerError, OSError, subprocess.SubprocessError) as exc:
        return (
            HealerResult(
                status=HealerStatus.ESCALATED,
                finding_id=finding.finding_id,
                investigation_reason=decision.investigation_reason,
                investigation_result="Automatic repair planning stopped fail-closed.",
                source_repo=state.config.healer.source_repo,
                before_screenshots=finding.screenshot_paths,
                api_evidence=finding.api_evidence,
                error=str(exc),
            ),
            None,
            UsageDelta(),
        )


async def heal_confirmed_finding(
    state: QAState, finding: CandidateFinding, decision: JudgeDecision
) -> tuple[HealerResult, UsageDelta]:
    """Convenience end-to-end execution used outside the staged LangGraph."""

    result, proposal, usage = await plan_confirmed_fix(state, finding, decision)
    if proposal is None or not result.worktree_path:
        return result, usage
    worktree = Path(result.worktree_path)
    try:
        changed = apply_validated_patch(worktree, proposal.unified_diff)
        commands, outputs = run_validation_profiles(worktree, proposal.validation_profiles)
        after_screenshots: list[str] = []
        if finding.kind == FindingKind.UI or "ui" in proposal.validation_profiles:
            after_screenshots, ui_output = await capture_after_ui_evidence(
                state, finding, worktree
            )
            commands.append("playwright-mcp: post-fix UI case")
            outputs.append(ui_output)
        validated = result.model_copy(
            update={
                "status": HealerStatus.VALIDATED,
                "investigation_result": (
                    f"Root cause: {proposal.root_cause}. Changed: {', '.join(changed)}"
                ),
                "validation_commands": commands,
                "validation_output": outputs,
                "after_screenshots": after_screenshots,
            }
        )
        return publish_draft_pr(state, finding, decision, validated, worktree), usage
    except (HealerError, OSError, subprocess.SubprocessError) as exc:
        return result.model_copy(
            update={
                "status": HealerStatus.ESCALATED,
                "investigation_result": "Automatic repair stopped fail-closed.",
                "error": str(exc),
            }
        ), usage
