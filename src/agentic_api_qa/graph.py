"""LangGraph topology exposed to the local Agent Server and LangSmith Studio."""

from langgraph.graph import END, START, StateGraph

from agentic_api_qa.models import QAState
from agentic_api_qa.nodes import (
    adversary_agent,
    adversary_http_tool,
    adversary_observe,
    explorer_agent,
    explorer_http_tool,
    explorer_observe,
    design_evaluator,
    governor,
    healer_agent,
    healer_patch_tool,
    healer_validation_tool,
    draft_pr_publisher,
    reviewer_security_tool,
    reviewer_agent,
    reviewer_pr_comment,
    healer_revision_agent,
    human_review_notifier,
    judge,
    reporter,
    playwright_mcp_tool,
    route_after_adversary_agent,
    route_after_adversary_observe,
    route_after_explorer_agent,
    route_after_explorer_observe,
    route_after_ui_explorer,
    route_after_ui_observe,
    route_after_judge,
    route_after_healer_agent,
    route_after_healer_patch,
    route_after_healer_validation,
    route_after_reviewer,
    ui_explorer_agent,
    ui_observe,
)


def build_graph():
    workflow = StateGraph(QAState)
    workflow.add_node("safety_governor", governor)
    workflow.add_node("explorer_agent", explorer_agent)
    workflow.add_node("explorer_http_tool", explorer_http_tool)
    workflow.add_node("explorer_observe", explorer_observe)
    workflow.add_node("adversary_agent", adversary_agent)
    workflow.add_node("adversary_http_tool", adversary_http_tool)
    workflow.add_node("adversary_observe", adversary_observe)
    workflow.add_node("ui_explorer_agent", ui_explorer_agent)
    workflow.add_node("playwright_mcp_tool", playwright_mcp_tool)
    workflow.add_node("ui_observe", ui_observe)
    workflow.add_node("design_evaluator", design_evaluator)
    workflow.add_node("judge", judge)
    workflow.add_node("healer_agent", healer_agent)
    workflow.add_node("healer_patch_tool", healer_patch_tool)
    workflow.add_node("healer_validation_tool", healer_validation_tool)
    workflow.add_node("draft_pr_publisher", draft_pr_publisher)
    workflow.add_node("reviewer_security_tool", reviewer_security_tool)
    workflow.add_node("reviewer_agent", reviewer_agent)
    workflow.add_node("healer_revision_agent", healer_revision_agent)
    workflow.add_node("reviewer_pr_comment", reviewer_pr_comment)
    workflow.add_node("human_review_notifier", human_review_notifier)
    workflow.add_node("json_reporter", reporter)

    workflow.add_edge(START, "safety_governor")
    workflow.add_edge("safety_governor", "explorer_agent")
    workflow.add_conditional_edges(
        "explorer_agent",
        route_after_explorer_agent,
        {
            "explorer_http_tool": "explorer_http_tool",
            "adversary_agent": "adversary_agent",
        },
    )
    workflow.add_edge("explorer_http_tool", "explorer_observe")
    workflow.add_conditional_edges(
        "explorer_observe",
        route_after_explorer_observe,
        {"explorer_agent": "explorer_agent", "adversary_agent": "adversary_agent"},
    )
    workflow.add_conditional_edges(
        "adversary_agent",
        route_after_adversary_agent,
        {
            "adversary_http_tool": "adversary_http_tool",
            "ui_explorer_agent": "ui_explorer_agent",
        },
    )
    workflow.add_edge("adversary_http_tool", "adversary_observe")
    workflow.add_conditional_edges(
        "adversary_observe",
        route_after_adversary_observe,
        {
            "adversary_agent": "adversary_agent",
            "ui_explorer_agent": "ui_explorer_agent",
        },
    )
    workflow.add_conditional_edges(
        "ui_explorer_agent",
        route_after_ui_explorer,
        {
            "playwright_mcp_tool": "playwright_mcp_tool",
            "design_evaluator": "design_evaluator",
        },
    )
    workflow.add_edge("playwright_mcp_tool", "ui_observe")
    workflow.add_conditional_edges(
        "ui_observe",
        route_after_ui_observe,
        {"ui_explorer_agent": "ui_explorer_agent"},
    )
    workflow.add_edge("design_evaluator", "judge")
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "healer_agent": "healer_agent",
            "human_review_notifier": "human_review_notifier",
            "json_reporter": "json_reporter",
        },
    )
    workflow.add_conditional_edges(
        "healer_agent",
        route_after_healer_agent,
        {
            "healer_patch_tool": "healer_patch_tool",
            "human_review_notifier": "human_review_notifier",
        },
    )
    workflow.add_conditional_edges(
        "healer_patch_tool",
        route_after_healer_patch,
        {
            "healer_validation_tool": "healer_validation_tool",
            "human_review_notifier": "human_review_notifier",
        },
    )
    workflow.add_conditional_edges(
        "healer_validation_tool",
        route_after_healer_validation,
        {
            "draft_pr_publisher": "draft_pr_publisher",
            "human_review_notifier": "human_review_notifier",
        },
    )
    workflow.add_edge("draft_pr_publisher", "reviewer_security_tool")
    workflow.add_edge("reviewer_security_tool", "reviewer_agent")
    workflow.add_conditional_edges(
        "reviewer_agent",
        route_after_reviewer,
        {
            "reviewer_pr_comment": "reviewer_pr_comment",
            "healer_revision_agent": "healer_revision_agent",
            "human_review_notifier": "human_review_notifier",
        },
    )
    workflow.add_conditional_edges(
        "healer_revision_agent",
        route_after_healer_agent,
        {
            "healer_patch_tool": "healer_patch_tool",
            "human_review_notifier": "human_review_notifier",
        },
    )
    workflow.add_edge("reviewer_pr_comment", "human_review_notifier")
    workflow.add_edge("human_review_notifier", "json_reporter")
    workflow.add_edge("json_reporter", END)
    return workflow.compile()


graph = build_graph()
