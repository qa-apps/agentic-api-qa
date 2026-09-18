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
    governor,
    judge,
    reporter,
    route_after_adversary_agent,
    route_after_adversary_observe,
    route_after_explorer_agent,
    route_after_explorer_observe,
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
    workflow.add_node("judge", judge)
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
        {"adversary_http_tool": "adversary_http_tool", "judge": "judge"},
    )
    workflow.add_edge("adversary_http_tool", "adversary_observe")
    workflow.add_conditional_edges(
        "adversary_observe",
        route_after_adversary_observe,
        {"adversary_agent": "adversary_agent", "judge": "judge"},
    )
    workflow.add_edge("judge", "json_reporter")
    workflow.add_edge("json_reporter", END)
    return workflow.compile()


graph = build_graph()
