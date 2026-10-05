from agentic_api_qa.ui_catalog import alexpavsky_ui_smoke_catalog


def test_ui_catalog_has_thirty_unique_safe_cases() -> None:
    cases = alexpavsky_ui_smoke_catalog()
    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30
    assert any(case.category == "chat" for case in cases)
    assert any(case.category == "authentication" for case in cases)
    assert any(case.category == "responsive" for case in cases)
    assert all(not any(action["tool"] == "browser_navigate" for action in case.actions) for case in cases)


def test_ui_actions_use_only_bounded_interactions() -> None:
    allowed = {"browser_click", "browser_fill_form", "browser_press_key"}
    assert {
        action["tool"]
        for case in alexpavsky_ui_smoke_catalog()
        for action in case.actions
    } <= allowed
