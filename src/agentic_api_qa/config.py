"""Environment-backed configuration helpers."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from agentic_api_qa.models import AgentModelConfig, Environment, RunConfig, SafetyLimits


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
            explorer_iterations=int(os.getenv("QA_EXPLORER_ITERATIONS", "4")),
            adversary_iterations=int(os.getenv("QA_ADVERSARY_ITERATIONS", "6")),
            explorer_tool_calls=int(os.getenv("QA_EXPLORER_TOOL_CALLS", "6")),
            adversary_tool_calls=int(os.getenv("QA_ADVERSARY_TOOL_CALLS", "10")),
            total_tool_calls=int(os.getenv("QA_TOTAL_TOOL_CALLS", "16")),
            request_timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "10")),
        ),
        models=AgentModelConfig(
            base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
            explorer_model=os.getenv("QA_EXPLORER_MODEL", "glm-5.3-flash"),
            adversary_model=os.getenv("QA_ADVERSARY_MODEL", "glm-5.3-flash"),
            reasoning_effort=os.getenv("QA_AGENT_REASONING_EFFORT", "high"),
            timeout_seconds=float(os.getenv("QA_LLM_TIMEOUT_SECONDS", "60")),
            max_tokens=int(os.getenv("QA_LLM_MAX_TOKENS", "1200")),
        ),
    )
