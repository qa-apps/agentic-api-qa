"""Typed access to the evaluation datasets and the frozen QAState fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_api_qa.models import QAState, Severity
from evals.config import DATASET_DIR, FIXTURE_DIR


AgentName = Literal["explorer", "adversary", "ui_explorer"]
ExpectedAction = Literal["execute", "finish", "any"]


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionScenario(EvalModel):
    """One frozen QAState plus the bounds its next agent decision must respect."""

    scenario_id: str
    agent: AgentName
    node: Literal["decide", "observe"] = "decide"
    fixture: str
    situation: str
    expected_behavior: str
    expected_action: ExpectedAction = "any"
    expected_result: str | None = None
    min_severity: Severity | None = None
    expected_continue_testing: bool | None = None
    acceptable_paths: list[str] = Field(default_factory=list)
    forbidden_paths: list[str] = Field(default_factory=list)
    acceptable_methods: list[str] = Field(default_factory=list)
    acceptable_effects: list[str] = Field(default_factory=list)
    acceptable_case_ids: list[str] = Field(default_factory=list)
    forbidden_case_ids: list[str] = Field(default_factory=list)
    require_investigation_reason: bool = False
    expected_stop_reason: str | None = None
    require_evidence_on_finish: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: ["next_action", "role_adherence", "priority"]
    )
    threshold: float | None = None

    def state(self) -> QAState:
        return load_qa_state(self.fixture)


class SeededBugProfile(EvalModel):
    """A closed bug manifest entry. Agents never receive this record."""

    profile_id: str
    title: str
    description: str
    mutations: list[str] = Field(min_length=1)
    surface: Literal["ui", "api", "both"]
    detection_agent: AgentName
    expected_kinds: list[Literal["ui", "api", "design"]] = Field(default_factory=list)
    match_any: list[str] = Field(default_factory=list)
    match_all: list[str] = Field(default_factory=list)
    expected_severity: Severity = Severity.HIGH
    severity_tolerance: int = Field(default=1, ge=0, le=4)
    max_actions: int = Field(default=10, ge=1, le=60)
    control: bool = False
    generated: bool = False

    @model_validator(mode="after")
    def seeded_profiles_need_detection_vocabulary(self) -> "SeededBugProfile":
        if not self.control and not (self.match_any and self.expected_kinds):
            raise ValueError(
                "A seeded profile needs expected_kinds and match_any; only a control "
                "profile may omit them."
            )
        return self


def _read_jsonl(path: Path) -> Iterator[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            yield json.loads(stripped)


def load_decision_scenarios(path: Path | None = None) -> list[DecisionScenario]:
    source = path or DATASET_DIR / "agent_decisions.jsonl"
    return [DecisionScenario.model_validate(item) for item in _read_jsonl(source)]


def load_seeded_profiles(path: Path | None = None) -> list[SeededBugProfile]:
    source = path or DATASET_DIR / "seeded_bugs.jsonl"
    return [SeededBugProfile.model_validate(item) for item in _read_jsonl(source)]


def load_qa_state(fixture: str) -> QAState:
    path = FIXTURE_DIR / fixture
    if not path.is_file():
        raise FileNotFoundError(f"Missing QAState fixture: {path}")
    return QAState.model_validate_json(path.read_text(encoding="utf-8"))
