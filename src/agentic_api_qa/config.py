"""Environment-backed configuration helpers."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from agentic_api_qa.models import (
    AgentModelConfig,
    Environment,
    HealerConfig,
    RunConfig,
    SafetyLimits,
)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_run_config() -> RunConfig:
    load_dotenv()
    return RunConfig(
        target_name=os.getenv("QA_TARGET_NAME", "alexpavsky"),
        base_url=os.getenv("API_BASE_URL", "https://www.alexpavsky.com"),
        environment=Environment(os.getenv("QA_ENVIRONMENT", "production")),
        production_read_only=_as_bool(os.getenv("QA_PRODUCTION_READ_ONLY"), True),
        require_mutation_approval=_as_bool(
            os.getenv("QA_REQUIRE_MUTATION_APPROVAL"), True
        ),
        limits=SafetyLimits(
            explorer_iterations=int(os.getenv("QA_EXPLORER_ITERATIONS", "30")),
            adversary_iterations=int(os.getenv("QA_ADVERSARY_ITERATIONS", "30")),
            explorer_tool_calls=int(os.getenv("QA_EXPLORER_TOOL_CALLS", "30")),
            adversary_tool_calls=int(os.getenv("QA_ADVERSARY_TOOL_CALLS", "30")),
            total_tool_calls=int(os.getenv("QA_TOTAL_TOOL_CALLS", "60")),
            request_timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "10")),
            soft_case_limit=int(os.getenv("QA_SOFT_CASE_LIMIT", "20")),
            hard_case_limit=int(os.getenv("QA_HARD_CASE_LIMIT", "30")),
            ui_tool_calls=int(os.getenv("QA_UI_TOOL_CALLS", "300")),
            healer_iterations=int(os.getenv("QA_HEALER_ITERATIONS", "8")),
            reviewer_revision_rounds=int(
                os.getenv("QA_REVIEWER_REVISION_ROUNDS", "2")
            ),
        ),
        models=AgentModelConfig(
            base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
            explorer_model=os.getenv("QA_EXPLORER_MODEL", "glm-5.3-flash"),
            adversary_model=os.getenv("QA_ADVERSARY_MODEL", "glm-5.3-flash"),
            ui_model=os.getenv("QA_UI_MODEL", "glm-5.3-flash"),
            design_model=os.getenv("QA_DESIGN_MODEL", "glm-5.3-flash"),
            judge_model=os.getenv("QA_JUDGE_MODEL", "glm-5.3-flash"),
            healer_model=os.getenv("QA_HEALER_MODEL", "glm-5.3-flash"),
            reviewer_model=os.getenv("QA_REVIEWER_MODEL", "glm-5.3-flash"),
            reasoning_effort=os.getenv("QA_AGENT_REASONING_EFFORT", "high"),
            judge_reasoning_effort=os.getenv("QA_JUDGE_REASONING_EFFORT", "max"),
            healer_reasoning_effort=os.getenv("QA_HEALER_REASONING_EFFORT", "high"),
            reviewer_reasoning_effort=os.getenv(
                "QA_REVIEWER_REASONING_EFFORT", "max"
            ),
            timeout_seconds=float(os.getenv("QA_LLM_TIMEOUT_SECONDS", "60")),
            max_tokens=int(os.getenv("QA_LLM_MAX_TOKENS", "1200")),
        ),
        healer=HealerConfig(
            enabled=_as_bool(os.getenv("QA_HEALER_ENABLED"), True),
            source_repo=os.getenv(
                "QA_HEALER_SOURCE_REPO", "/Users/alex/Projects/alexpavsky"
            ),
            remote=os.getenv("QA_HEALER_REMOTE", "origin"),
            base_branch=os.getenv("QA_HEALER_BASE_BRANCH", "main"),
            github_repository=os.getenv(
                "QA_HEALER_GITHUB_REPOSITORY", "qa-apps/alexpavsky"
            ),
            worktree_root=os.getenv("QA_HEALER_WORKTREE_ROOT", ".healer-worktrees"),
            create_draft_pr=_as_bool(os.getenv("QA_HEALER_DRAFT_PR"), True),
            minimum_judge_confidence=float(
                os.getenv("QA_JUDGE_MIN_CONFIDENCE", "0.85")
            ),
            human_review_channel_id=os.getenv("QA_HUMAN_REVIEW_CHANNEL_ID") or None,
        ),
        ui_enabled=_as_bool(os.getenv("QA_UI_ENABLED"), True),
        ui_headless=_as_bool(os.getenv("QA_UI_HEADLESS"), True),
        ui_video=os.getenv("QA_UI_VIDEO", "retain-on-failure"),
        ui_artifact_dir=os.getenv("QA_UI_ARTIFACT_DIR", "artifacts/ui"),
    )
