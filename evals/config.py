"""Environment-backed settings for the evaluation layer.

The evaluation judge must not be the model under test, so ``judge_model`` defaults to
a stronger Z.AI model than the ``glm-5.3-flash`` the agents run on.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(__file__).resolve().parent / "datasets"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "qa_states"


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in {None, ""} else None


class EvalConfig(BaseModel):
    """Evaluation settings; secrets stay in the environment."""

    model_config = ConfigDict(extra="forbid")

    judge_model: str = "glm-5.3"
    judge_base_url: str = "https://api.z.ai/api/paas/v4/"
    judge_reasoning_effort: str = "high"
    judge_max_tokens: int = Field(default=2_000, ge=256, le=16_384)
    judge_timeout_seconds: float = Field(default=120, gt=0, le=600)
    agent_input_usd_per_million: float | None = Field(default=None, ge=0)
    agent_output_usd_per_million: float | None = Field(default=None, ge=0)
    judge_input_usd_per_million: float | None = Field(default=None, ge=0)
    judge_output_usd_per_million: float | None = Field(default=None, ge=0)

    deepeval_enabled: bool = True
    decision_threshold: float = Field(default=0.6, ge=0, le=1)
    finding_threshold: float = Field(default=0.6, ge=0, le=1)

    site_source: Path = Path("/Users/alex/Projects/alexpavsky")
    site_host: str = "127.0.0.1"
    site_port: int = Field(default=0, ge=0, le=65_535)

    seeded_max_ui_cases: int = Field(default=8, ge=1, le=30)
    seeded_max_api_cases: int = Field(default=6, ge=1, le=30)
    seeded_ui_enabled: bool = True

    mutation_seed: int = 1337
    mutation_count: int = Field(default=0, ge=0, le=5)

    slack_enabled: bool = True
    slack_channel_id: str | None = "C0BPQTUS2DR"
    report_dir: Path = REPO_ROOT / "reports" / "evals"


def load_eval_config() -> EvalConfig:
    load_dotenv()
    return EvalConfig(
        judge_model=os.getenv("QA_EVAL_JUDGE_MODEL", "glm-5.3"),
        judge_base_url=os.getenv(
            "QA_EVAL_JUDGE_BASE_URL",
            os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
        ),
        judge_reasoning_effort=os.getenv("QA_EVAL_JUDGE_REASONING_EFFORT", "high"),
        judge_max_tokens=int(os.getenv("QA_EVAL_JUDGE_MAX_TOKENS", "2000")),
        judge_timeout_seconds=float(os.getenv("QA_EVAL_JUDGE_TIMEOUT_SECONDS", "120")),
        agent_input_usd_per_million=_optional_float(
            os.getenv("QA_EVAL_AGENT_INPUT_USD_PER_MILLION")
        ),
        agent_output_usd_per_million=_optional_float(
            os.getenv("QA_EVAL_AGENT_OUTPUT_USD_PER_MILLION")
        ),
        judge_input_usd_per_million=_optional_float(
            os.getenv("QA_EVAL_JUDGE_INPUT_USD_PER_MILLION")
        ),
        judge_output_usd_per_million=_optional_float(
            os.getenv("QA_EVAL_JUDGE_OUTPUT_USD_PER_MILLION")
        ),
        deepeval_enabled=_as_bool(os.getenv("QA_EVAL_DEEPEVAL"), True),
        decision_threshold=float(os.getenv("QA_EVAL_DECISION_THRESHOLD", "0.6")),
        finding_threshold=float(os.getenv("QA_EVAL_FINDING_THRESHOLD", "0.6")),
        site_source=Path(
            os.getenv("QA_EVAL_SITE_SOURCE", "/Users/alex/Projects/alexpavsky")
        ),
        site_host=os.getenv("QA_EVAL_SITE_HOST", "127.0.0.1"),
        site_port=int(os.getenv("QA_EVAL_SITE_PORT", "0")),
        seeded_max_ui_cases=int(os.getenv("QA_EVAL_SEEDED_UI_CASES", "8")),
        seeded_max_api_cases=int(os.getenv("QA_EVAL_SEEDED_API_CASES", "6")),
        seeded_ui_enabled=_as_bool(os.getenv("QA_EVAL_SEEDED_UI_ENABLED"), True),
        mutation_seed=int(os.getenv("QA_EVAL_MUTATION_SEED", "1337")),
        mutation_count=int(os.getenv("QA_EVAL_MUTATION_COUNT", "0")),
        slack_enabled=_as_bool(os.getenv("QA_EVAL_SLACK_ENABLED"), True),
        slack_channel_id=os.getenv("QA_EVAL_SLACK_CHANNEL_ID", "C0BPQTUS2DR") or None,
        report_dir=Path(os.getenv("QA_EVAL_REPORT_DIR", str(REPO_ROOT / "reports" / "evals"))),
    )


def agent_api_key_available() -> bool:
    load_dotenv()
    return bool(os.getenv("ZAI_API_KEY"))
