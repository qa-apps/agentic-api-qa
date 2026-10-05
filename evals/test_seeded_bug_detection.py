"""End-to-end seeded-bug detection against a local mutated copy of the site.

The agents never see the manifest. They receive an ordinary localhost site, run the full
graph, and their findings are compared with the closed bug profile afterwards.
"""

from __future__ import annotations

import pytest

from evals.dataset import SeededBugProfile, load_seeded_profiles
from evals.harness import run_seeded_profile
from evals.seeded_site import MUTATIONS
from evals.seeded_site.server import SiteSourceError, load_sources


PROFILES = load_seeded_profiles()


def _ids(profiles: list[SeededBugProfile]) -> list[str]:
    return [profile.profile_id for profile in profiles]


@pytest.fixture(scope="session", autouse=True)
def require_site_sources(eval_config):
    try:
        load_sources(eval_config)
    except SiteSourceError as exc:
        pytest.skip(str(exc))


def test_every_profile_uses_known_mutations() -> None:
    unknown = {
        mutation
        for profile in PROFILES
        for mutation in profile.mutations
        if mutation != "none" and mutation not in MUTATIONS
    }

    assert not unknown, f"unknown mutations in the manifest: {sorted(unknown)}"


@pytest.mark.seeded_bugs
@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
@pytest.mark.asyncio
async def test_agents_detect_the_seeded_bug(
    profile: SeededBugProfile, eval_config, judge
) -> None:
    outcome = await run_seeded_profile(
        profile, config=eval_config, judge=judge, use_deepeval=judge is not None
    )

    assert outcome.error is None, outcome.error
    assert not outcome.policy_violations, outcome.policy_violations
    if profile.control:
        assert not outcome.false_positives, (
            "findings were invented on the clean control copy: "
            f"{outcome.false_positives}"
        )
        return
    assert outcome.detected, (
        f"{profile.profile_id} was not reported. "
        f"Findings: {outcome.reported_findings}, tool calls: {outcome.tool_calls}, "
        f"false positives: {outcome.false_positives}"
    )
    assert outcome.within_budget, (
        f"detected after {outcome.actions_to_detect} actions, "
        f"budget was {profile.max_actions}"
    )
    assert outcome.evidence_grounding > 0, "the matching finding carries no evidence"
