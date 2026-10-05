"""Slack human-review handoff with evidence attachments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agentic_api_qa.models import HumanReviewRequest, QAState
from agentic_api_qa.observability import langfuse_trace_url


class HumanReviewNotificationError(RuntimeError):
    pass


async def _slack_api(
    client: httpx.AsyncClient, token: str, method: str, *, data: dict
) -> dict:
    response = await client.post(
        f"https://slack.com/api/{method}",
        headers={"Authorization": f"Bearer {token}"},
        data=data,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise HumanReviewNotificationError(
            f"Slack {method} failed: {body.get('error', 'unknown_error')}"
        )
    return body


async def _upload_file(
    client: httpx.AsyncClient,
    token: str,
    channel_id: str,
    thread_ts: str,
    path: Path,
) -> None:
    content = path.read_bytes()
    prepared = await _slack_api(
        client,
        token,
        "files.getUploadURLExternal",
        data={"filename": path.name, "length": str(len(content))},
    )
    upload = await client.post(
        prepared["upload_url"],
        files={"filename": (path.name, content)},
    )
    upload.raise_for_status()
    await _slack_api(
        client,
        token,
        "files.completeUploadExternal",
        data={
            "files": json.dumps([{"id": prepared["file_id"], "title": path.name}]),
            "channel_id": channel_id,
            "thread_ts": thread_ts,
        },
    )


def _review_text(state: QAState) -> str:
    healer = state.healer_result
    confirmed = [
        decision
        for decision in state.judge_decisions
        if decision.verdict.value == "CONFIRMED"
    ]
    lines = [
        ":rotating_light: *Human review required — AP Agentic QA*",
        f"*Run:* `{state.run_id}`",
        f"*Confirmed findings:* {len(confirmed)}",
    ]
    if healer is not None:
        lines.extend(
            [
                f"*Healer status:* `{healer.status.value}`",
                f"*Investigation reason:* {healer.investigation_reason or 'n/a'}",
                f"*Investigation result:* {healer.investigation_result or 'n/a'}",
                f"*Proposed fix:* {healer.proposed_fix or 'No automatic patch; review evidence.'}",
                f"*Draft PR:* {healer.pr_url or 'not created'}",
            ]
        )
        if healer.validation_commands:
            lines.append(
                "*Validation:* "
                + ", ".join(f"`{command}`" for command in healer.validation_commands)
            )
        if healer.error:
            lines.append(f"*Fail-closed reason:* {healer.error}")
    reviewer = state.reviewer_decision
    scan = state.reviewer_scan
    if reviewer is not None:
        lines.extend(
            [
                f"*Agent Reviewer:* `{reviewer.verdict.value}` ({reviewer.confidence:.0%})",
                f"*Reviewed commit:* `{reviewer.approved_commit_sha or 'not approved'}`",
                f"*Security:* {reviewer.security_assessment}",
                f"*Deletion/scope:* {reviewer.deletion_assessment}",
                f"*Code quality:* {reviewer.quality_assessment}",
            ]
        )
        if reviewer.required_changes:
            lines.append("*Required changes:* " + "; ".join(reviewer.required_changes))
    if scan is not None:
        lines.append(
            "*Automated gates:* "
            f"secrets={scan.secret_scan_passed}, static={scan.static_scan_passed}, "
            f"dependencies={scan.dependency_scan_passed}, deletions={scan.deletion_guard_passed}"
        )
    trace = langfuse_trace_url(state.run_id)
    if trace:
        lines.append(f"*Langfuse trace:* {trace}")
    lines.append(":no_entry_sign: *No merge was performed. Human approval is mandatory.*")
    return "\n".join(lines)


async def notify_human_review(state: QAState) -> HumanReviewRequest:
    """Post one parent message, then attach all available before/after screenshots."""

    load_dotenv()
    channel_id = state.config.healer.human_review_channel_id
    finding_ids = [item.finding_id for item in state.confirmed_findings]
    healer = state.healer_result
    evidence_paths = list(
        dict.fromkeys(
            [
                *(
                    healer.before_screenshots + healer.after_screenshots
                    if healer is not None
                    else []
                ),
                *[
                    path
                    for finding in state.confirmed_findings
                    for path in finding.screenshot_paths
                ],
            ]
        )
    )
    request = HumanReviewRequest(
        required=bool(finding_ids),
        reason=(
            "Judge confirmed one or more findings; every patch requires human review."
            if finding_ids
            else ""
        ),
        finding_ids=finding_ids,
        pr_url=healer.pr_url if healer else None,
        slack_channel_id=channel_id,
        evidence_paths=evidence_paths,
    )
    if not request.required:
        return request
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token or not channel_id:
        return request.model_copy(
            update={
                "notification_error": (
                    "SLACK_BOT_TOKEN or QA_HUMAN_REVIEW_CHANNEL_ID is not configured; "
                    "the review request remains in the JSON/HTML report."
                )
            }
        )
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            posted = await _slack_api(
                client,
                token,
                "chat.postMessage",
                data={"channel": channel_id, "text": _review_text(state)},
            )
            thread_ts = str(posted["ts"])
            for raw_path in evidence_paths:
                path = Path(raw_path)
                if path.is_file():
                    await _upload_file(client, token, channel_id, thread_ts, path)
        return request.model_copy(update={"slack_message_ts": thread_ts})
    except (httpx.HTTPError, OSError, KeyError, HumanReviewNotificationError) as exc:
        return request.model_copy(update={"notification_error": str(exc)})
