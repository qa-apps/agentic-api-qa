from pathlib import Path

import httpx
import pytest

from agentic_api_qa.models import CandidateFinding, FindingKind, Severity
from evals.config import EvalConfig, load_eval_config
from evals.dataset import load_decision_scenarios, load_seeded_profiles
from evals.metrics import matches_profile
from evals.seeded_site import serve_profile
from evals.seeded_site.server import article_url_rejection


def test_evaluation_datasets_and_fixtures_are_valid() -> None:
    scenarios = load_decision_scenarios()
    profiles = load_seeded_profiles()

    assert 10 <= len(scenarios) + len(profiles) <= 20
    assert len({item.scenario_id for item in scenarios}) == len(scenarios)
    assert len({item.profile_id for item in profiles}) == len(profiles)
    assert all(item.state().run_id.startswith("eval-") for item in scenarios)
    assert any(item.control for item in profiles)


@pytest.fixture
def site_config(tmp_path: Path) -> EvalConfig:
    """A minimal site checkout, so the API contract tests run without the real app."""

    (tmp_path / "index.html").write_text(
        "<!doctype html><html><head><title>Test site</title>"
        '<link rel="stylesheet" href="/style.css"></head>'
        '<body><main><section id="feed"><div id="feed-grid"></div></section></main>'
        '<script src="/script.js"></script></body></html>',
        encoding="utf-8",
    )
    (tmp_path / "style.css").write_text("body { margin: 0; }", encoding="utf-8")
    (tmp_path / "script.js").write_text(
        "var base = isLocalPreview ? 'http://127.0.0.1:8000' : '';", encoding="utf-8"
    )
    (tmp_path / "article-reader.js").write_text("window.reader = true;", encoding="utf-8")
    (tmp_path / "assets" / "vendor").mkdir(parents=True)
    (tmp_path / "assets" / "vendor" / "purify.min.js").write_text("/* purify */", encoding="utf-8")
    (tmp_path / "assets" / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / "chat_server.py").write_text("SECRET = 1", encoding="utf-8")
    return load_eval_config().model_copy(update={"site_source": tmp_path})


def test_seeded_site_keeps_the_manifest_hidden_and_applies_api_bugs(
    site_config: EvalConfig,
) -> None:
    config = site_config
    profiles = {item.profile_id: item for item in load_seeded_profiles()}

    with serve_profile(profiles["chat-api-500"], config) as site:
        response = httpx.post(
            f"{site.base_url}/api/chat", json={"message": "Hello"}, timeout=5
        )
        homepage = httpx.get(site.base_url, timeout=5)

    assert response.status_code == 500
    assert "chat-api-500" not in homepage.text
    assert "seeded_bugs" not in homepage.text


def test_clean_control_copy_returns_healthy_contract_responses(
    site_config: EvalConfig,
) -> None:
    config = site_config
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


def test_clean_control_copy_serves_every_page_resource_same_origin(
    site_config: EvalConfig,
) -> None:
    control = next(item for item in load_seeded_profiles() if item.control)

    with serve_profile(control, site_config) as site:
        homepage = httpx.get(site.base_url, timeout=5)
        script = httpx.get(f"{site.base_url}/script.js", timeout=5)
        reader = httpx.get(f"{site.base_url}/article-reader.js?v=1", timeout=5)
        vendor = httpx.get(f"{site.base_url}/assets/vendor/purify.min.js", timeout=5)
        image = httpx.get(f"{site.base_url}/assets/logo.png", timeout=5)
        source = httpx.get(f"{site.base_url}/chat_server.py", timeout=5)
        feed = httpx.get(f"{site.base_url}/api/feed", timeout=5)

    assert "Cross-origin request disabled in the test copy" in homepage.text
    assert "127.0.0.1:8000" not in script.text
    assert reader.status_code == vendor.status_code == image.status_code == 200
    assert source.status_code == 404
    articles = feed.json()["articles"]
    assert len(articles) >= 3
    assert all({"title", "link", "date", "category"} <= set(item) for item in articles)


def test_broken_images_profile_blocks_only_images(site_config: EvalConfig) -> None:
    profile = next(item for item in load_seeded_profiles() if item.profile_id == "broken-images")

    with serve_profile(profile, site_config) as site:
        image = httpx.get(f"{site.base_url}/assets/logo.png", timeout=5)
        vendor = httpx.get(f"{site.base_url}/assets/vendor/purify.min.js", timeout=5)

    assert image.status_code == 404
    assert vendor.status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:1/private",
        "http://169.254.169.254/latest/meta-data/",
        "http://2130706433/",
        "http://[::1]/",
        "http://localhost/",
        "http://db.internal/",
        "file:///etc/passwd",
    ],
)
def test_article_proxy_rejects_private_targets_like_the_application(url: str) -> None:
    assert article_url_rejection(url) is not None


def test_clean_control_copy_rejects_ssrf_and_serves_public_articles(
    site_config: EvalConfig,
) -> None:
    control = next(item for item in load_seeded_profiles() if item.control)

    with serve_profile(control, site_config) as site:
        loopback = httpx.get(
            f"{site.base_url}/api/article-proxy",
            params={"url": "http://127.0.0.1:1/private"},
            timeout=5,
        )
        public = httpx.get(
            f"{site.base_url}/api/article-proxy",
            params={"url": "https://93.184.216.34/article"},
            timeout=5,
        )

    assert loopback.status_code == 400
    assert public.status_code == 200
    assert public.json()["title"]


def test_clean_control_copy_remembers_each_chat_session(site_config: EvalConfig) -> None:
    control = next(item for item in load_seeded_profiles() if item.control)

    with serve_profile(control, site_config) as site:
        chat = f"{site.base_url}/api/chat"
        first = httpx.post(chat, json={"message": "Explain flaky tests", "session_id": "a"}, timeout=5)
        follow_up = httpx.post(chat, json={"message": "Summarize that", "session_id": "a"}, timeout=5)
        other = httpx.post(chat, json={"message": "Summarize that", "session_id": "b"}, timeout=5)

    assert "Explain flaky tests" not in first.json()["reply"]
    assert "Explain flaky tests" in follow_up.json()["reply"]
    assert "Explain flaky tests" not in other.json()["reply"]


def test_clean_control_copy_rejects_a_non_string_chat_message(site_config: EvalConfig) -> None:
    control = next(item for item in load_seeded_profiles() if item.control)

    with serve_profile(control, site_config) as site:
        response = httpx.post(
            f"{site.base_url}/api/chat", json={"message": {"text": "hi"}}, timeout=5
        )

    assert response.status_code == 400


def test_missing_news_feed_profile_also_empties_the_feed_api(site_config: EvalConfig) -> None:
    profile = next(
        item for item in load_seeded_profiles() if item.profile_id == "missing-news-feed"
    )

    with serve_profile(profile, site_config) as site:
        feed = httpx.get(f"{site.base_url}/api/feed", timeout=5)

    assert feed.json() == {"articles": []}
