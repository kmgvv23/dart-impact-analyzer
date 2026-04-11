"""LangGraph StateGraph — Supervisor 분기 + 파이프라인."""

from langgraph.graph import END, START, StateGraph

from disclosure_agent.nodes import (
    classify_rules_node,
    early_exit_node,
    ingest_node,
    llm_judge_node,
    load_dart_node,
    multi_batch_node,
    multi_report_writer_node,
    price_pattern_node,
    report_writer_node,
    route_after_classify,
    route_after_judge,
    route_after_load,
    price_context_llm_node,
    similar_cases_llm_node,
)
from disclosure_agent.state import AgentState


def build_disclosure_graph():
    g = StateGraph(AgentState)

    g.add_node("ingest", ingest_node)
    g.add_node("load_dart", load_dart_node)
    g.add_node("classify_rules", classify_rules_node)
    g.add_node("llm_judge", llm_judge_node)
    g.add_node("similar_cases", similar_cases_llm_node)
    g.add_node("price_full", lambda s: price_pattern_node(s, light=False))
    g.add_node("price_context", price_context_llm_node)
    g.add_node("report_writer", report_writer_node)
    g.add_node("multi_batch", multi_batch_node)
    g.add_node("multi_report_writer", multi_report_writer_node)
    g.add_node("early_exit", early_exit_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "load_dart")
    g.add_conditional_edges(
        "load_dart",
        route_after_load,
        {"early_exit": "early_exit", "classify_rules": "classify_rules", "multi_batch": "multi_batch"},
    )
    g.add_conditional_edges(
        "classify_rules",
        route_after_classify,
        {"early_exit": "early_exit", "llm_judge": "llm_judge"},
    )
    g.add_conditional_edges(
        "llm_judge",
        route_after_judge,
        {"early_exit": "early_exit", "similar_cases": "similar_cases"},
    )
    g.add_edge("similar_cases", "price_full")
    g.add_edge("price_full", "price_context")
    g.add_edge("price_context", "report_writer")
    g.add_edge("report_writer", END)
    g.add_edge("multi_batch", "multi_report_writer")
    g.add_edge("multi_report_writer", END)
    g.add_edge("early_exit", END)

    return g.compile()
