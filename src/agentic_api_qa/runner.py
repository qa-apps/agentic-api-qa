"""CLI entry point for the complete Agentic API QA graph."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentic_api_qa.config import load_run_config
from agentic_api_qa.graph import graph
from agentic_api_qa.models import Environment, QAState


async def run_workflow(state: QAState | None = None) -> QAState:
    initial = state or QAState(config=load_run_config())
    result = await graph.ainvoke(initial)
    return QAState.model_validate(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentic API QA workflow")
    parser.add_argument("--base-url", help="Override API_BASE_URL")
    parser.add_argument(
        "--environment",
        choices=[item.value for item in Environment],
        help="Target environment safety profile",
    )
    parser.add_argument("--output", type=Path, help="JSON report destination")
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    config = load_run_config()
    updates = {}
    if args.base_url:
        updates["base_url"] = args.base_url
    if args.environment:
        updates["environment"] = Environment(args.environment)
    if updates:
        config = config.model_copy(update=updates)
    state = await run_workflow(QAState(config=config))
    if state.report is None:
        raise RuntimeError("Workflow completed without a report")
    output = args.output or Path("reports") / f"qa-report-{state.run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(state.report.model_dump_json(indent=2), encoding="utf-8")
    print(state.report.summary)
    print(f"Report: {output.resolve()}")
    return 0 if state.report.run.overall_result.value == "PASS" else 1


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()

