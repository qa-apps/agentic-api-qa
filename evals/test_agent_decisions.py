"""Evaluate the next decision a real agent makes from a frozen QAState.

Deterministic assertions own the mechanics (tool, parameters, origin, budget, repeats,
finish, evidence). The DeepEval judge owns decision quality.
"""

from __future__ import annotations

import pytest

from evals.dataset import DecisionScenario, load_decision_scenarios
from evals.harness import run_decision_scenario


SCENARIOS = load_decision_scenarios()


def _ids(scenarios: list[DecisionScenario]) -> list[str]:
    return [scenario.scenario_id for scenario in scenarios]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
@pytest.mark.asyncio
async def test_agent_decision_respects_the_deterministic_contract(
    scenario: DecisionScenario, eval_config
) -> None:
    outcome = await run_decision_scenario(scenario, config=eval_config, use_deepeval=False)

    assert outcome.error is None, outcome.error
    assert not outcome.violations, (
        f"{scenario.scenario_id}: {outcome.violations}\n"
        f"decision: {outcome.action} {outcome.case_id} — {outcome.summary}"
    )


@pytest.mark.parametrize(
    "scenario",
    [scenario for scenario in SCENARIOS if scenario.metrics],
    ids=_ids([scenario for scenario in SCENARIOS if scenario.metrics]),
)
@pytest.mark.asyncio
async def test_agent_decision_quality_meets_the_judge_threshold(
    scenario: DecisionScenario, eval_config, judge
) -> None:
    if judge is None:
        pytest.skip("DeepEval scoring is disabled (QA_EVAL_DEEPEVAL=0)")
    outcome = await run_decision_scenario(
        scenario, config=eval_config, judge=judge, use_deepeval=True
    )

    assert outcome.error is None, outcome.error
    assert outcome.judge_scores, "the judge returned no scores"
    judge_errors = [item for item in outcome.judge_scores if item.error]
    assert not judge_errors, [item.error for item in judge_errors]
    failed = [item for item in outcome.judge_scores if not item.passed]
    assert not failed, "\n".join(
        f"{item.name}: {item.score:.2f} < {item.threshold:.2f} — {item.reason}"
        for item in failed
    )
