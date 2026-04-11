from typing import Annotated, Any, Literal, Optional, TypedDict

AnalysisMode = Literal["single", "multi"]

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


ImpactLevel = Literal["HIGH", "MEDIUM", "LOW"]


class AgentState(TypedDict, total=False):
    """seed.yaml ontology + LangGraph 메시지."""

    disclosure_input: str
    corp_name_hint: str
    disclosure_title: str
    rcept_no: str
    disclosure_pick_reason: str
    stock_code: str
    corp_code: str
    corp_name: str
    disclosure_date: str
    raw_body: str
    disclosure_type: str
    extracted_fields: dict[str, Any]
    llm_stance: str
    impact_level: ImpactLevel
    early_exit_reason: str
    grey_zone: bool
    peer_set: list[dict[str, Any]]
    peer_search_note: str
    similar_cases_narrative: str
    price_pattern: dict[str, Any]
    price_insight_narrative: str
    final_report: dict[str, Any]
    analysis_mode: AnalysisMode
    disclosure_queue: list[dict[str, Any]]
    batch_results: list[dict[str, Any]]
    messages: Annotated[list[BaseMessage], add_messages]
    react_thread_id: str
