import json

from agentic_api_qa.client import _normalize_json_body, _serialize


def test_json_body_given_as_text_is_decoded_once() -> None:
    """An agent emitting the *text* of a JSON object must not double-encode it.

    Passing the raw string to httpx's json= would put a JSON string on the
    wire, which the target then reports as malformed input.
    """
    decoded = _normalize_json_body('{"message": "ping"}')

    assert decoded == {"message": "ping"}
    assert json.loads(_serialize(decoded)) == {"message": "ping"}


def test_json_body_given_as_array_text_is_decoded() -> None:
    assert _normalize_json_body("[1, 2, 3]") == [1, 2, 3]


def test_structured_json_body_is_left_alone() -> None:
    body = {"message": "ping", "history": []}

    assert _normalize_json_body(body) is body


def test_plain_string_body_is_not_treated_as_json() -> None:
    """A bare string is a deliberate scalar payload, not a document to decode."""
    assert _normalize_json_body("ping") == "ping"


def test_json_scalar_text_is_not_promoted_to_a_document() -> None:
    """'12345' parses as JSON but is not an object or array, so it stays text."""
    assert _normalize_json_body("12345") == "12345"


def test_malformed_json_text_is_passed_through_untouched() -> None:
    assert _normalize_json_body("{nope") == "{nope"


def test_none_body_is_preserved() -> None:
    assert _normalize_json_body(None) is None
