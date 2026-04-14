"""기능 2 — 재무·공시 질의용 ReAct 에이전트 (LangGraph prebuilt)."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from disclosure_agent.config import get_openai_api_key
from disclosure_agent.tools import CHAT_TOOLS


def build_finance_react_agent():
    model = ChatOpenAI(
        model="gpt-5-mini",
        api_key=get_openai_api_key(),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 한국 상장사 DART·시장 맥락 도우미입니다. "
                "필요하면 dart_financials, dart_disclosure_search, web_search 도구를 사용하세요. "
                "법인코드(corp_code)가 없으면 사용자에게 DART 8자리 법인코드 또는 정확한 법인명을 물어보세요. "
                "투자 권유는 하지 말고 수치·출처 위주로 답합니다.",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    return create_react_agent(model, CHAT_TOOLS, prompt=prompt)
