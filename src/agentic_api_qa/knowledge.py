"""Load a small, relevant QA knowledge slice for each agent decision."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentic_api_qa.models import Actor, QAState


KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "knowledge"


@lru_cache(maxsize=1)
def load_target_contract() -> dict[str, Any]:
    return json.loads((KNOWLEDGE_ROOT / "target_contract.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load_jsonl(name: str) -> tuple[dict[str, Any], ...]:
    path = KNOWLEDGE_ROOT / "examples" / name
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def select_reference_examples(
    state: QAState, actor: Actor, *, limit: int = 5
) -> list[dict[str, Any]]:
    """Rotate through curated examples so prompts stay compact and non-prescriptive."""

    primary_name = (
        "api_positive.jsonl" if actor == Actor.EXPLORER else "api_adversarial.jsonl"
    )
    primary = list(_load_jsonl(primary_name))
    false_positives = list(_load_jsonl("false_positives.jsonl"))
    iteration = (
        state.explorer_iterations if actor == Actor.EXPLORER else state.adversary_iterations
    )
    if primary:
        offset = iteration % len(primary)
        primary = primary[offset:] + primary[:offset]
    selected = primary[: max(1, limit - 1)]
    selected.append(false_positives[iteration % len(false_positives)])
    return selected[:limit]


def agent_knowledge(state: QAState, actor: Actor) -> dict[str, Any]:
    contract = load_target_contract()
    return {
        "contract_purpose": contract["purpose"],
        "global_validations": contract["global_validations"],
        "endpoints": contract["endpoints"],
        "test_data_policy": contract["test_data_policy"],
        "reference_examples": select_reference_examples(state, actor),
        "reference_rule": (
            "Examples illustrate validation quality only. Do not copy them as a fixed "
            "plan; choose the next action from current evidence and safety constraints."
        ),
    }
