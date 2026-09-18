from pydantic import ValidationError
import pytest

from agentic_api_qa.models import RequestSpec, SafetyLimits


def test_request_rejects_two_body_representations() -> None:
    with pytest.raises(ValidationError):
        RequestSpec(
            operation_id="invalid",
            method="POST",
            path="/api/example",
            json_body={"hello": "world"},
            raw_body="hello",
        )


def test_safety_limits_are_bounded() -> None:
    with pytest.raises(ValidationError):
        SafetyLimits(explorer_iterations=21)
