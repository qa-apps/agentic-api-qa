"""Evaluation-suite setup. Paid LLM evaluations never run inside the fast pytest suite."""

from __future__ import annotations

import os

import pytest

from evals.config import agent_api_key_available, load_eval_config


# Evaluation runs keep their own report directory and never spawn the visual report server.
os.environ.setdefault("QA_REPORT_DIR", "reports/evals/runs")
os.environ.setdefault("QA_REPORT_SERVER_ENABLED", "false")


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        item.add_marker(pytest.mark.evals)


@pytest.fixture(scope="session")
def eval_config():
    return load_eval_config()


@pytest.fixture(scope="session", autouse=True)
def require_agent_credentials():
    if not agent_api_key_available():
        pytest.skip("ZAI_API_KEY is required to evaluate real agent decisions")


@pytest.fixture(scope="session")
def judge(eval_config):
    """One shared judge model, so token accounting and cost stay visible per session."""

    if not eval_config.deepeval_enabled:
        return None
    pytest.importorskip("deepeval", reason="pip install -e '.[evals]' to score decisions")
    from evals.judges import build_judge

    return build_judge(eval_config)
