"""CLI orchestrator for the evaluation layer.

Examples
--------
Fast, unpaid contract check of every decision scenario::

    python -m evals.run_evals --suite decisions --no-deepeval

Full nightly run with judge scoring, seeded bugs, and reproducible random mutations::

    python -m evals.run_evals --suite all --mutations 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from typing import Any

os.environ.setdefault("QA_REPORT_DIR", "reports/evals/runs")
os.environ.setdefault("QA_REPORT_SERVER_ENABLED", "false")

from evals.config import agent_api_key_available, load_eval_config  # noqa: E402
from evals.dataset import (  # noqa: E402
    DecisionScenario,
    SeededBugProfile,
    load_decision_scenarios,
    load_seeded_profiles,
)
from evals.harness import run_decision_scenario, run_seeded_profile  # noqa: E402
from evals.metrics import EvalRunReport, build_report  # noqa: E402
from evals.reporting import (  # noqa: E402
    notify_eval_failure,
    record_langfuse_scores,
    render_markdown,
    write_report,
)
from evals.seeded_site.bugs import random_profiles  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the agent evaluation suites")
    parser.add_argument(
        "--suite",
        choices=["decisions", "seeded", "all"],
        default="all",
        help="Which evaluations to run",
    )
    parser.add_argument("--scenario", action="append", help="Limit to these scenario ids")
    parser.add_argument("--profile", action="append", help="Limit to these seeded profile ids")
    parser.add_argument(
        "--mutations",
        type=int,
        default=None,
        help="Additionally run N reproducible random mutations as a non-blocking report",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random mutation seed")
    parser.add_argument(
        "--no-deepeval", action="store_true", help="Skip paid LLM-judge scoring"
    )
    parser.add_argument("--no-slack", action="store_true", help="Never post Slack alerts")
    parser.add_argument("--label", help="Explicit run label used in report file names")
    return parser


async def _run_decisions(
    scenarios: list[DecisionScenario], *, settings, judge: Any | None, label: str
) -> list:
    outcomes = []
    for scenario in scenarios:
        outcome = await run_decision_scenario(
            scenario,
            config=settings,
            judge=judge,
            use_deepeval=judge is not None,
            run_label=label,
        )
        status = "pass" if outcome.passed else "FAIL"
        print(
            f"[decision] {scenario.scenario_id}: {status} "
            f"action={outcome.action} violations={len(outcome.violations)} "
            f"judge={outcome.deepeval_score:.2f}"
        )
        outcomes.append(outcome)
    return outcomes


async def _run_seeded(
    profiles: list[SeededBugProfile], *, settings, judge: Any | None
) -> list:
    outcomes = []
    for profile in profiles:
        outcome = await run_seeded_profile(
            profile, config=settings, judge=judge, use_deepeval=judge is not None
        )
        status = "pass" if outcome.passed else "FAIL"
        print(
            f"[seeded] {profile.profile_id}: {status} detected={outcome.detected} "
            f"actions={outcome.actions_to_detect} tool_calls={outcome.tool_calls} "
            f"fps={len(outcome.false_positives)} judge={outcome.deepeval_score:.2f}"
        )
        outcomes.append(outcome)
    return outcomes


def _publish(report: EvalRunReport, *, settings, slack: bool) -> None:
    paths = write_report(report, settings)
    print(render_markdown(report))
    print(f"Report: {paths['markdown']}")
    published = record_langfuse_scores(report)
    if published:
        print(f"Langfuse scores published: {published}")
    if slack and not report.passed:
        error = notify_eval_failure(report, settings)
        print(f"Slack alert: {error or 'sent'}")


async def _main() -> int:
    args = _parser().parse_args()
    settings = load_eval_config()
    if args.no_deepeval:
        settings = settings.model_copy(update={"deepeval_enabled": False})
    if args.seed is not None:
        settings = settings.model_copy(update={"mutation_seed": args.seed})
    if not agent_api_key_available():
        print("ZAI_API_KEY is not configured; real agents cannot be evaluated.")
        return 2

    judge = None
    if settings.deepeval_enabled:
        from evals.judges import build_judge

        judge = build_judge(settings)
        print(f"Judge model: {judge.get_model_name()}")

    label = args.label or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    started = datetime.now(timezone.utc)
    scenarios = load_decision_scenarios()
    profiles = load_seeded_profiles()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [item for item in scenarios if item.scenario_id in wanted]
    if args.profile:
        wanted = set(args.profile)
        profiles = [item for item in profiles if item.profile_id in wanted]

    decisions = (
        await _run_decisions(scenarios, settings=settings, judge=judge, label=label)
        if args.suite in {"decisions", "all"}
        else []
    )
    seeded = (
        await _run_seeded(profiles, settings=settings, judge=judge)
        if args.suite in {"seeded", "all"}
        else []
    )
    report = build_report(
        suite=args.suite,
        run_label=label,
        started_at=started,
        judge_model=judge.get_model_name() if judge else "deterministic-only",
        decisions=decisions,
        seeded=seeded,
    )
    _publish(report, settings=settings, slack=not args.no_slack)

    mutation_count = args.mutations if args.mutations is not None else settings.mutation_count
    if mutation_count:
        print(
            f"\nRandomized non-blocking run: {mutation_count} mutations, "
            f"seed {settings.mutation_seed}"
        )
        random_started = datetime.now(timezone.utc)
        random_outcomes = await _run_seeded(
            random_profiles(seed=settings.mutation_seed, count=mutation_count),
            settings=settings,
            judge=judge,
        )
        random_report = build_report(
            suite="random-mutations",
            run_label=f"{label}-random",
            started_at=random_started,
            judge_model=judge.get_model_name() if judge else "deterministic-only",
            decisions=[],
            seeded=random_outcomes,
            mutation_seed=settings.mutation_seed,
            blocking=False,
        )
        _publish(random_report, settings=settings, slack=not args.no_slack)

    return 0 if report.passed else 1


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
