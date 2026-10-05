"""Evaluation reporting: local artifacts, Langfuse scores, and Slack failure alerts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import load_dotenv

from agentic_api_qa.observability import (
    flush_langfuse,
    langfuse_enabled,
    langfuse_trace_id,
)
from evals.config import EvalConfig, load_eval_config
from evals.metrics import DecisionOutcome, EvalRunReport, SeededOutcome


SLACK_API = "https://slack.com/api/chat.postMessage"


def record_eval_observation(
    *,
    run_id: str,
    name: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Attach the post-run evidence and evaluation result to the agent trace."""

    if not langfuse_enabled():
        return
    from langfuse import get_client

    client = get_client()
    observation = client.start_observation(
        trace_context={"trace_id": langfuse_trace_id(run_id)},
        name=name,
        as_type="evaluator",
        input=input_data,
        output=output_data,
        metadata={"run_id": run_id, **metadata},
    )
    observation.end()


def _scores_for_seeded(outcome: SeededOutcome) -> list[tuple[str, float, str]]:
    scores: list[tuple[str, float, str]] = [
        ("eval_bug_detected", float(outcome.detected), "BOOLEAN"),
        ("eval_judge_confirmed", float(outcome.judge_confirmed), "BOOLEAN"),
        ("eval_policy_violations", float(len(outcome.policy_violations)), "NUMERIC"),
        ("eval_false_positives", float(len(outcome.false_positives)), "NUMERIC"),
        ("eval_evidence_grounding", outcome.evidence_grounding, "NUMERIC"),
        ("eval_tool_calls", float(outcome.tool_calls), "NUMERIC"),
    ]
    if outcome.actions_to_detect is not None:
        scores.append(("eval_time_to_detect", float(outcome.actions_to_detect), "NUMERIC"))
    if outcome.severity_accuracy is not None:
        scores.append(("eval_severity_accuracy", outcome.severity_accuracy, "NUMERIC"))
    if outcome.judge_scores:
        scores.append(("eval_deepeval_score", outcome.deepeval_score, "NUMERIC"))
    return scores


def _scores_for_decision(outcome: DecisionOutcome) -> list[tuple[str, float, str]]:
    scores: list[tuple[str, float, str]] = [
        ("eval_decision_passed", float(outcome.passed), "BOOLEAN"),
        ("eval_policy_violations", float(len(outcome.violations)), "NUMERIC"),
    ]
    if outcome.judge_scores:
        scores.append(("eval_deepeval_score", outcome.deepeval_score, "NUMERIC"))
        scores.extend(
            (f"eval_{item.name.lower().replace(' ', '_').replace('-', '_')}", item.score, "NUMERIC")
            for item in outcome.judge_scores
            if item.error is None
        )
    return scores


def record_langfuse_scores(report: EvalRunReport) -> int:
    """Attach evaluation scores to the Langfuse trace of every scored agent run."""

    if not langfuse_enabled():
        return 0
    from langfuse import get_client

    client = get_client()
    published = 0
    groups: list[tuple[str, list[tuple[str, float, str]], dict[str, Any]]] = []
    for outcome in report.seeded:
        if not outcome.run_id:
            continue
        groups.append(
            (
                outcome.run_id,
                _scores_for_seeded(outcome),
                {
                    "eval_suite": report.suite,
                    "eval_run": report.run_label,
                    "profile_id": outcome.profile_id,
                    "judge_model": outcome.judge_model or report.judge_model,
                },
            )
        )
    for outcome in report.decisions:
        if not outcome.run_id:
            continue
        groups.append(
            (
                outcome.run_id,
                _scores_for_decision(outcome),
                {
                    "eval_suite": report.suite,
                    "eval_run": report.run_label,
                    "scenario_id": outcome.scenario_id,
                    "agent": outcome.agent,
                    "judge_model": outcome.judge_model or report.judge_model,
                },
            )
        )
    for run_id, scores, metadata in groups:
        trace_id = langfuse_trace_id(run_id)
        for name, value, data_type in scores:
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=cast(Any, data_type),
                metadata=metadata,
            )
            published += 1
    flush_langfuse()
    return published


def render_markdown(report: EvalRunReport) -> str:
    metrics = report.metrics
    lines = [
        f"# Agent Evaluation `{report.run_label}`",
        "",
        f"Suite: {report.suite}",
        f"Judge model: {report.judge_model}",
        f"Started: {report.started_at.isoformat(timespec='seconds')}",
        f"Finished: {report.finished_at.isoformat(timespec='seconds')}",
        f"Result: {'PASS' if report.passed else 'FAIL'}"
        + ("" if report.blocking else " (non-blocking)"),
        "",
        "## Metrics",
        "",
        f"- Bug recall: {metrics.bug_recall:.0%} "
        f"({metrics.bugs_detected}/{metrics.seeded_profiles})",
        f"- Judge-confirmed recall: {metrics.judge_confirmed_recall:.0%}",
        f"- False-positive rate: {metrics.false_positive_rate:.0%}",
        f"- Mean time-to-detect: "
        + (
            f"{metrics.mean_time_to_detect:.1f} actions"
            if metrics.mean_time_to_detect is not None
            else "n/a"
        ),
        f"- Mean tool calls: {metrics.mean_tool_calls:.1f}",
        f"- Policy violation rate: {metrics.policy_violation_rate:.0%}",
        f"- Evidence grounding: {metrics.evidence_grounding:.0%}",
        f"- Severity accuracy: "
        + (
            f"{metrics.severity_accuracy:.0%}"
            if metrics.severity_accuracy is not None
            else "n/a"
        ),
        f"- Decision pass rate: {metrics.decision_pass_rate:.0%} "
        f"({metrics.decision_scenarios} scenarios)",
        f"- DeepEval score: {metrics.deepeval_score:.2f}",
        f"- Tokens: {metrics.total_tokens}",
        f"- Estimated cost: "
        + (
            f"${metrics.estimated_cost_usd:.6f}"
            if metrics.estimated_cost_usd is not None
            else "n/a (configure per-token rates)"
        ),
        f"- Wall time: {metrics.total_latency_ms / 1000:.1f}s",
    ]
    if report.decisions:
        lines.extend(
            [
                "",
                "## Agent decisions",
                "",
                "| Scenario | Agent | Action | Violations | DeepEval | Result |",
                "| -------- | ----- | ------ | ---------- | -------- | ------ |",
            ]
        )
        for item in report.decisions:
            lines.append(
                f"| {item.scenario_id} | {item.agent}/{item.node} | {item.action} | "
                f"{'; '.join(item.violations) or 'none'} | "
                f"{item.deepeval_score:.2f} | {'PASS' if item.passed else 'FAIL'} |"
            )
    if report.seeded:
        lines.extend(
            [
                "",
                "## Seeded bugs",
                "",
                "| Profile | Detected | Confirmed | Actions | Tool calls | FPs | DeepEval | Result |",
                "| ------- | -------- | --------- | ------- | ---------- | --- | -------- | ------ |",
            ]
        )
        for item in report.seeded:
            lines.append(
                f"| {item.profile_id} | {'yes' if item.detected else 'no'} | "
                f"{'yes' if item.judge_confirmed else 'no'} | "
                f"{item.actions_to_detect if item.actions_to_detect is not None else '-'}"
                f"/{item.max_actions} | {item.tool_calls} | {len(item.false_positives)} | "
                f"{item.deepeval_score:.2f} | {'PASS' if item.passed else 'FAIL'} |"
            )
    failures = report.failures
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {name}" for name in failures)
    return "\n".join(lines) + "\n"


def write_report(report: EvalRunReport, config: EvalConfig | None = None) -> dict[str, Path]:
    settings = config or load_eval_config()
    destination = Path(settings.report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"eval-{report.run_label}.json"
    markdown_path = destination / f"eval-{report.run_label}.md"
    payload = report.model_dump_json(indent=2)
    markdown = render_markdown(report)
    json_path.write_text(payload, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_stem = "latest" if report.blocking else "latest-random"
    (destination / f"{latest_stem}.json").write_text(payload, encoding="utf-8")
    (destination / f"{latest_stem}.md").write_text(markdown, encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _failure_blocks(report: EvalRunReport) -> list[str]:
    blocks: list[str] = []
    for item in report.seeded:
        if item.passed:
            continue
        if item.error:
            actual = f"run failed: {item.error}"
        elif item.control and item.false_positives:
            actual = f"invented {len(item.false_positives)} finding(s) on a clean copy"
        elif not item.detected:
            actual = "bug not reported"
        elif item.policy_violations:
            actual = f"policy violated: {'; '.join(item.policy_violations)}"
        elif any(not score.passed for score in item.judge_scores):
            actual = (
                f"DeepEval quality score {item.deepeval_score:.2f} is below the "
                "configured threshold"
            )
        else:
            actual = (
                f"detected after {item.actions_to_detect} actions, "
                f"budget was {item.max_actions}"
            )
        blocks.append(
            "\n".join(
                [
                    "❌ *AP Agents · Evaluation failed*",
                    f"*Profile:* {item.profile_id}",
                    f"*Expected:* {item.detection_expectation}",
                    f"*Actual:* {actual}",
                    f"*Tool calls:* {item.tool_calls}",
                    f"*DeepEval score:* {item.deepeval_score:.2f}",
                    f"*Langfuse trace:* {item.langfuse_trace_url or 'not configured'}",
                ]
            )
        )
    for item in report.decisions:
        if item.passed:
            continue
        actual = item.error or "; ".join(item.violations) or "judge score below threshold"
        blocks.append(
            "\n".join(
                [
                    "❌ *AP Agents · Evaluation failed*",
                    f"*Scenario:* {item.scenario_id}",
                    f"*Agent:* {item.agent} ({item.node})",
                    f"*Actual:* {actual}",
                    f"*Decision:* {item.action} {item.case_id or ''}".strip(),
                    f"*DeepEval score:* {item.deepeval_score:.2f}",
                ]
            )
        )
    return blocks


def failure_message(report: EvalRunReport) -> str:
    header = (
        f"❌ *AP Agents · Evaluation failed* — `{report.run_label}` ({report.suite})\n"
        f"Bug recall {report.metrics.bug_recall:.0%} | "
        f"policy violations {report.metrics.policy_violation_rate:.0%} | "
        f"DeepEval {report.metrics.deepeval_score:.2f}"
        + ("" if report.blocking else "\n_Non-blocking randomized run._")
    )
    return "\n\n".join([header, *_failure_blocks(report)])[:38_000]


def notify_eval_failure(
    report: EvalRunReport, config: EvalConfig | None = None
) -> str | None:
    """Post one Slack alert for a failed evaluation run. Returns an error string or None."""

    settings = config or load_eval_config()
    if report.passed:
        return None
    if not settings.slack_enabled or not settings.slack_channel_id:
        return "Slack alerting is disabled for evaluations"
    load_dotenv()
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return "SLACK_BOT_TOKEN is not configured; the failure stays in the local report"
    try:
        response = httpx.post(
            SLACK_API,
            headers={"Authorization": f"Bearer {token}"},
            data={
                "channel": settings.slack_channel_id,
                "text": failure_message(report),
            },
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        return f"Slack request failed: {type(exc).__name__}: {exc}"
    if not body.get("ok"):
        return f"Slack rejected the alert: {body.get('error', 'unknown_error')}"
    return None
