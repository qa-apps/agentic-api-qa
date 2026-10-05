"""DeepEval judge backed by Z.AI, running a different model than the agents.

DeepEval defaults to OpenAI. This project only carries a Z.AI key, so the judge is a
custom ``DeepEvalBaseLLM`` pinned to a stronger GLM model than the ``glm-5.3-flash`` the
Explorer, Adversary, and UI agents decide with. Using the model under test as its own
judge would make the evaluation self-confirming, so that is refused by default.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from agentic_api_qa.config import load_run_config
from evals.config import EvalConfig, load_eval_config


class EvalJudgeError(RuntimeError):
    """Raised when the judge model cannot produce a usable evaluation."""


def _agent_models() -> set[str]:
    models = load_run_config().models
    return {
        models.explorer_model,
        models.adversary_model,
        models.ui_model,
        models.judge_model,
    }


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise EvalJudgeError("Judge returned no JSON object") from None
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise EvalJudgeError("Judge returned invalid JSON") from exc


class ZaiJudgeModel:
    """Custom DeepEval LLM. Kept import-light so plain pytest never needs DeepEval."""

    def __init__(self, config: EvalConfig | None = None) -> None:
        load_dotenv()
        self.config = config or load_eval_config()
        self.name = self.config.judge_model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        allow_same = os.getenv("QA_EVAL_ALLOW_SAME_MODEL", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not allow_same and self.name in _agent_models():
            raise EvalJudgeError(
                f"Judge model {self.name!r} is also used by the agents under test. "
                "Set QA_EVAL_JUDGE_MODEL to a different model."
            )
        self.model = self.load_model()

    # DeepEvalBaseLLM interface -------------------------------------------------
    def load_model(self) -> "ZaiJudgeModel":
        return self

    def get_model_name(self) -> str:
        return f"z.ai/{self.name}"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        return self._parse(self._request(prompt, schema is not None), schema)

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        return self._parse(await self._arequest(prompt, schema is not None), schema)

    def supports_json_mode(self) -> bool:
        return True

    # Transport ----------------------------------------------------------------
    def _payload(self, prompt: str, json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge for an autonomous QA agent "
                        "system. Judge only the supplied evidence, never reward confident "
                        "prose, and answer exactly in the requested format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": self.config.judge_max_tokens,
            "thinking": {"type": "enabled"},
            "reasoning_effort": self.config.judge_reasoning_effort,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _endpoint(self) -> str:
        return f"{self.config.judge_base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv("ZAI_API_KEY")
        if not api_key:
            raise EvalJudgeError("ZAI_API_KEY is not configured for the eval judge")
        return {"Authorization": f"Bearer {api_key}"}

    def _request(self, prompt: str, json_mode: bool) -> str:
        try:
            response = httpx.post(
                self._endpoint(),
                headers=self._headers(),
                json=self._payload(prompt, json_mode),
                timeout=self.config.judge_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EvalJudgeError(f"Judge request failed: {type(exc).__name__}: {exc}") from exc
        return self._content(response.json())

    async def _arequest(self, prompt: str, json_mode: bool) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.config.judge_timeout_seconds) as client:
                response = await client.post(
                    self._endpoint(),
                    headers=self._headers(),
                    json=self._payload(prompt, json_mode),
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EvalJudgeError(f"Judge request failed: {type(exc).__name__}: {exc}") from exc
        return self._content(response.json())

    def _content(self, data: dict[str, Any]) -> str:
        usage = data.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EvalJudgeError("Judge response contained no message") from exc
        if not content:
            raise EvalJudgeError("Judge returned empty content")
        return str(content)

    def _parse(self, content: str, schema: type[BaseModel] | None) -> Any:
        if schema is None:
            return content
        return schema.model_validate(_extract_json(content))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def build_judge(config: EvalConfig | None = None) -> ZaiJudgeModel:
    """Return a DeepEval-compatible judge, inheriting DeepEvalBaseLLM when available."""

    try:
        from deepeval.models import DeepEvalBaseLLM
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise EvalJudgeError(
            "DeepEval is not installed. Install the optional extra: pip install -e '.[evals]'"
        ) from exc

    class _Judge(ZaiJudgeModel, DeepEvalBaseLLM):
        pass

    return _Judge(config)
