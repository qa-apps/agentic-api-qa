import httpx

from agentic_api_qa.models import CandidateFinding, FindingKind, Severity
from evals.config import load_eval_config
from evals.dataset import load_decision_scenarios, load_seeded_profiles
from evals.metrics import matches_profile
from evals.seeded_site import serve_profile


def test_evaluation_datasets_and_fixtures_are_valid() -> None:
    scenarios = load_decision_scenarios()
    profiles = load_seeded_profiles()

    assert 10 <= len(scenarios) + len(profiles) <= 20
    assert len({item.scenario_id for item in scenarios}) == len(scenarios)
    assert len({item.profile_id for item in profiles}) == len(profiles)
    assert all(item.state().run_id.startswith("eval-") for item in scenarios)
    assert any(item.control for item in profiles)


def test_seeded_site_keeps_the_manifest_hidden_and_applies_api_bugs() -> None:
    config = load_eval_config()
    profiles = {item.profile_id: item for item in load_seeded_profiles()}

    with serve_profile(profiles["chat-api-500"], config) as site:
        response = httpx.post(
            f"{site.base_url}/api/chat", json={"message": "Hello"}, timeout=5
        )
        homepage = httpx.get(site.base_url, timeout=5)

    assert response.status_code == 500
    assert "chat-api-500" not in homepage.text
    assert "seeded_bugs" not in homepage.text


def test_clean_control_copy_returns_healthy_contract_responses() -> None:
    config = load_eval_config()
    control = next(item for item in load_seeded_profiles() if item.control)

    with serve_profile(control, config) as site:
        health = httpx.get(f"{site.base_url}/api/health", timeout=5)
        feed = httpx.get(f"{site.base_url}/api/feed", timeout=5)
        chat = httpx.post(
            f"{site.base_url}/api/chat", json={"message": "Hello"}, timeout=5
        )

    assert health.status_code == 200
    assert feed.status_code == 200
    assert chat.status_code == 200
    assert "system prompt:" not in chat.text.casefold()


def test_closed_manifest_matches_only_relevant_findings() -> None:
    profile = next(
        item for item in load_seeded_profiles() if item.profile_id == "chat-api-500"
    )
    relevant = CandidateFinding(
        source="explorer",
        kind=FindingKind.API,
        severity=Severity.HIGH,
        title="Chat API returned HTTP 500",
        expected="A valid chat request returns a reply.",
        actual="POST /api/chat returned an internal server error.",
    )
    unrelated = CandidateFinding(
        source="ui_explorer",
        kind=FindingKind.UI,
        severity=Severity.LOW,
        title="Theme color differs",
        expected="Theme is consistent.",
        actual="A button uses a different blue.",
    )

    assert matches_profile(relevant, profile)
    assert not matches_profile(unrelated, profile)
