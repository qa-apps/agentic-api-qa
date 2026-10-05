"""Post the nightly AP agents run summary to Slack.

The full run lives in the published HTML report; this message is the daily pointer to
it. It says which agents ran, what they covered and what the judge confirmed, so a
reader can tell this run apart from the layout-only design check in the same channel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from agentic_api_qa.models import FinalReport, JudgeVerdict, OverallResult

SCOPE = (
    "Explorer, Adversary and UI agents test alexpavsky.com end to end: API contract, "
    "adversarial guardrails and UI journeys. An LLM judge confirms every finding "
    "before it counts. Production runs are read-only."
)
_STATUS = {
    OverallResult.PASS: ("✅", "PASS", "#2eb886"),
    OverallResult.FAIL: ("❌", "FAIL", "#cc2929"),
    OverallResult.ERROR: ("⚠️", "ERROR", "#e9a820"),
}
_MAX_LISTED_FINDINGS = 5


def _confirmed_findings(report: FinalReport) -> list[tuple[str, str]]:
    confirmed = {
        decision.finding_id
        for decision in report.judge_decisions
        if decision.verdict == JudgeVerdict.CONFIRMED
    }
    return [
        (finding.severity.value, finding.title)
        for finding in report.candidate_findings
        if finding.finding_id in confirmed
    ]


def build_payload(
    report: FinalReport,
    *,
    channel: str,
    report_url: str = "",
    run_url: str = "",
) -> dict[str, Any]:
    emoji, label, color = _STATUS[report.run.overall_result]
    metrics = report.metrics
    confirmed = _confirmed_findings(report)
    ui_total = metrics.ui_cases_passed + metrics.ui_cases_failed
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji}  AP Agents · Agentic QA  —  {label}",
                "emoji": True,
            },
        },
        {"type": "context", "elements": [{"type": "mrkdwn", "text": SCOPE}]},
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*API happy path*\n{metrics.happy_path_passed} passed · "
                        f"{metrics.happy_path_failed} failed"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Guardrails (Adversary)*\n{metrics.guardrails_held} held · "
                        f"{metrics.guardrails_breached} breached · "
                        f"{metrics.guardrails_inconclusive} inconclusive"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*UI journeys*\n{metrics.ui_cases_passed}/{ui_total} passed"
                        if ui_total
                        else "*UI journeys*\nnot run"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Judge-confirmed findings*\n{len(confirmed)} of "
                        f"{len(report.candidate_findings)} candidates"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Agent effort*\n{metrics.total_tool_calls} tool calls · "
                        f"{metrics.llm_total_tokens:,} tokens"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Target*\n{report.run.target} ({report.run.environment.value}) · "
                        f"{report.run.duration_ms / 1000:.0f}s"
                    ),
                },
            ],
        },
    ]
    if confirmed:
        lines = [f"• [{severity}] {title}" for severity, title in confirmed[:_MAX_LISTED_FINDINGS]]
        if len(confirmed) > _MAX_LISTED_FINDINGS:
            lines.append(f"…and {len(confirmed) - _MAX_LISTED_FINDINGS} more in the full report")
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Confirmed findings*\n" + "\n".join(lines)},
            }
        )
    if report.pipeline_errors:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Pipeline errors:* {len(report.pipeline_errors)} — "
                        f"{report.pipeline_errors[0].message[:200]}"
                    ),
                },
            }
        )
    buttons: list[dict[str, Any]] = []
    if report_url:
        buttons.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Open full report", "emoji": True},
                "url": report_url,
                "style": "primary" if report.run.overall_result == OverallResult.PASS else "danger",
            }
        )
    if run_url:
        buttons.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "GitHub run", "emoji": True},
                "url": run_url,
            }
        )
    if buttons:
        blocks.append({"type": "actions", "elements": buttons})
    return {
        "channel": channel,
        "text": f"AP Agents · Agentic QA — {label}",
        "attachments": [{"color": color, "blocks": blocks}],
    }


def post(payload: dict[str, Any], token: str) -> str | None:
    """Return the Slack error code, or None when the message was accepted."""

    response = httpx.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=15,
    )
    body = response.json()
    return None if body.get("ok") else str(body.get("error", "unknown_error"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="FinalReport JSON")
    parser.add_argument("--channel", required=True, help="Slack channel ID")
    parser.add_argument("--report-url", default="", help="Published full report URL")
    parser.add_argument("--run-url", default="", help="GitHub Actions run URL")
    args = parser.parse_args(argv)

    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        print("SLACK_BOT_TOKEN is not set; skipping the Slack summary.", file=sys.stderr)
        return 0
    report = FinalReport.model_validate(json.loads(args.report.read_text(encoding="utf-8")))
    payload = build_payload(
        report, channel=args.channel, report_url=args.report_url, run_url=args.run_url
    )
    error = post(payload, token)
    if error:
        print(f"Slack rejected the summary: {error}", file=sys.stderr)
        return 1
    print(f"Posted the run summary to {args.channel}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
