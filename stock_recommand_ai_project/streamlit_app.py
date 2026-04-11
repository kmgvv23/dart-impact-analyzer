"""
한국 공시 투자 판단 에이전트 — Streamlit UI (seed.yaml / PRD).
실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv

# disclosure_agent import 전에 환경 변수 로드
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from disclosure_agent.chat_agent import build_finance_react_agent
from disclosure_agent.graph import build_disclosure_graph


@st.cache_resource
def get_disclosure_graph():
    return build_disclosure_graph()


@st.cache_resource
def get_finance_react():
    return build_finance_react_agent()


def _build_disclosure_input(company: str, disclosure_link: str) -> str:
    """ingest: 접수/URL이 있으면 첫 줄=링크, 둘째 줄=기업명 힌트. 기업명만이면 회사명 검색."""
    c = (company or "").strip()
    u = (disclosure_link or "").strip()
    if u and c:
        return f"{u}\n{c}"
    if u:
        return u
    return c


def _impact_style(level: str) -> tuple[str, str, str]:
    lv = (level or "LOW").upper()
    if lv == "HIGH":
        return ("높음", "#b91c1c", "#fff1f2")
    if lv == "MEDIUM":
        return ("보통", "#b45309", "#fffbeb")
    return ("낮음", "#374151", "#f3f4f6")


def _render_impact_banner(level: str, subtitle: str = "") -> None:
    label_ko, fg, bg = _impact_style(level)
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:16px;margin:0 0 8px 0;padding:16px 20px;
border-radius:12px;background:{bg};border:1px solid {fg}33;box-shadow:0 1px 2px rgba(0,0,0,0.06);">
  <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.06em;color:{fg};">이번 공시 영향</div>
  <div style="font-size:1.75rem;font-weight:800;color:{fg};line-height:1;">{label_ko}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


def _batch_impact_counts(batch_results: list) -> dict[str, int]:
    c = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for b in batch_results or []:
        if b.get("skipped"):
            continue
        k = str(b.get("impact_level") or "LOW").upper()
        if k in c:
            c[k] += 1
    return c


def _split_legacy_rationale(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""
    m = re.search(
        r"(?:^|\n)\s*(?:부정적(?:/불확실)?\s*요인|불리한\s*점|리스크\s*요인)\s*[:：]\s*",
        raw,
        re.I,
    )
    if m:
        pos = raw[: m.start()].strip()
        neg = raw[m.end() :].strip()
        pos = re.sub(
            r"^\s*(?:긍정적\s*요인|긍정\s*요인|장점)\s*[:：]\s*",
            "",
            pos,
            count=1,
            flags=re.I,
        ).strip()
        return pos, neg
    return raw, ""


def _legacy_scenarios(s: str) -> tuple[str, str, str]:
    s = (s or "").strip()
    if not s:
        return "", "", ""
    markers: list[tuple[int, int, str]] = []
    for name, pat in [
        ("base", r"(?i)(?:^|\n)\s*베이스\s*시나리오\s*[:：]\s*"),
        ("best", r"(?i)(?:^|\n)\s*(?:베스트|최선)\s*시나리오\s*[:：]\s*"),
        ("worst", r"(?i)(?:^|\n)\s*(?:워스트|최악)\s*시나리오\s*[:：]\s*"),
    ]:
        m = re.search(pat, s)
        if m:
            markers.append((m.start(), m.end(), name))
    markers.sort(key=lambda x: x[0])
    if not markers:
        return s, "", ""
    parts: dict[str, str] = {"base": "", "best": "", "worst": ""}
    preamble = s[: markers[0][0]].strip()
    for i, (_st, en, name) in enumerate(markers):
        nend = markers[i + 1][0] if i + 1 < len(markers) else len(s)
        parts[name] = s[en:nend].strip()
    if preamble:
        parts["base"] = (preamble + ("\n\n" if parts["base"] else "") + parts["base"]).strip()
    return parts["base"], parts["best"], parts["worst"]


def _render_final_report(rep: dict) -> None:
    if not rep:
        return
    st.markdown("### 리포트")
    st.markdown(rep.get("summary") or "")

    pos = (rep.get("impact_positive") or "").strip()
    neg = (rep.get("impact_negative") or "").strip()
    legacy_r = (rep.get("impact_rationale") or "").strip()
    if not pos and not neg and legacy_r:
        pos, neg = _split_legacy_rationale(legacy_r)

    if pos or neg:
        st.markdown("#### 영향 근거")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**유리한 점**")
            st.markdown(pos or "—")
        with c2:
            st.markdown("**불리·불확실**")
            st.markdown(neg or "—")

    ppd = (rep.get("peer_price_digest") or "").strip()
    if ppd:
        st.markdown("#### 주가·맥락 요약")
        st.markdown(ppd)

    sb = (rep.get("scenario_base") or "").strip()
    sbe = (rep.get("scenario_best") or "").strip()
    sw = (rep.get("scenario_worst") or "").strip()
    leg_sc = (rep.get("scenario") or "").strip()
    if not (sb or sbe or sw) and leg_sc:
        sb, sbe, sw = _legacy_scenarios(leg_sc)
    if sb or sbe or sw:
        st.markdown("#### 시나리오")
        if sb:
            st.markdown("##### Base")
            st.markdown(sb)
        if sbe:
            st.markdown("##### Best")
            st.markdown(sbe)
        if sw:
            st.markdown("##### Worst")
            st.markdown(sw)

    acts = rep.get("actions") or {}
    if acts:
        st.markdown("#### 참고 한 줄")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.caption("손절")
        ac1.write(acts.get("손절", "-"))
        ac2.caption("익절")
        ac2.write(acts.get("익절", "-"))
        ac3.caption("진입")
        ac3.write(acts.get("진입", "-"))
        ac4.caption("관망")
        ac4.write(acts.get("관망", "-"))
    if rep.get("disclaimer"):
        st.caption(rep["disclaimer"])


def main():
    st.set_page_config(page_title="공시 영향 노트", layout="wide", initial_sidebar_state="collapsed")
    st.title("공시 영향 노트")
    st.caption("공시가 시장에 미칠 만한 영향과 주가 맥락을 짧게 정리합니다. 투자 결정은 본인 책임이에요.")

    tab1, tab2 = st.tabs(["공시 분석", "질문하기"])

    with tab1:
        st.markdown(
            "회사 이름만 넣으면 최근 공시 중 눈에 띄는 건들을 묶어 보여 주고, "
            "**공시 화면 주소**를 함께 넣으면 그 한 건만 집중해서 봅니다."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            company_in = st.text_input("회사 이름", placeholder="예: 삼성전자", key="corp_in")
        with col_b:
            link_in = st.text_input(
                "특정 공시 주소(선택)",
                placeholder="공시 페이지 URL을 붙여넣기",
                key="link_in",
            )

        if st.button("분석하기", type="primary", key="run_disclosure"):
            disclosure_input = _build_disclosure_input(company_in, link_in)
            if not disclosure_input.strip():
                st.warning("회사 이름이나 공시 주소 중 하나는 입력해 주세요.")
            else:
                with st.spinner("분석 중…"):
                    try:
                        graph = get_disclosure_graph()
                        out = graph.invoke({"disclosure_input": disclosure_input})
                    except Exception as e:
                        st.error(f"실행 오류: {e}")
                        return

                mode = out.get("analysis_mode") or "single"
                il = str(out.get("impact_level") or "LOW").upper()
                if mode == "multi":
                    br = out.get("batch_results") or []
                    cnt = _batch_impact_counts(br)
                    sub = (
                        f"최근 공시 {len(br)}건을 살펴봤어요. "
                        f"영향 높음 {cnt['HIGH']} · 보통 {cnt['MEDIUM']} · 낮음 {cnt['LOW']} — "
                        f"위 배지는 그중 가장 중요한 한 건 기준이에요."
                    )
                else:
                    sub = (out.get("disclosure_title") or "")[:120]
                _render_impact_banner(il, sub)

                pr = (out.get("disclosure_pick_reason") or "").strip()
                if pr:
                    st.caption(pr)

                meta_cols = st.columns(4)
                meta_cols[0].metric("분석 범위", "여러 공시" if mode == "multi" else "한 건")
                meta_cols[1].metric("공시유형", str(out.get("disclosure_type") or "-"))
                meta_cols[2].metric("접수번호", str(out.get("rcept_no") or "-"))
                meta_cols[3].metric("종목코드", str(out.get("stock_code") or "-"))

                rep = out.get("final_report") or {}
                st.markdown("---")
                _render_final_report(rep)

                st.markdown("---")
                st.caption("더 보기")

                sim = (out.get("similar_cases_narrative") or "").strip()
                if sim or out.get("peer_search_note"):
                    with st.expander("유사 기업·유형 참고", expanded=False):
                        if out.get("peer_search_note"):
                            st.caption(out["peer_search_note"])
                        if sim:
                            st.markdown(sim)
                        elif il in ("HIGH", "MEDIUM"):
                            st.caption("참고 사례 문장을 가져오지 못했습니다. 잠시 후 다시 실행해 보시거나 상세 영역을 확인해 주세요.")
                        else:
                            st.caption("이번 분석에서는 참고 사례 단계가 포함되지 않았습니다.")

                pi = (out.get("price_insight_narrative") or "").strip()
                if pi or out.get("price_pattern"):
                    with st.expander("시세·주가 흐름 (상세)", expanded=False):
                        st.caption("최근 종가·수익률 등은 거래소 공개 시세를 바탕으로 정리한 초안입니다.")
                        if pi:
                            st.markdown(pi)
                        else:
                            st.json(out.get("price_pattern") or {})

                with st.expander("부가 정보", expanded=False):
                    st.json(
                        {
                            "analysis_mode": mode,
                            "rcept_no": out.get("rcept_no"),
                            "disclosure_title": out.get("disclosure_title"),
                            "disclosure_type": out.get("disclosure_type"),
                            "impact_level": out.get("impact_level"),
                            "grey_zone": out.get("grey_zone"),
                            "corp_name": out.get("corp_name"),
                            "stock_code": out.get("stock_code"),
                            "peer_search_note": out.get("peer_search_note"),
                            "similar_cases_preview": (out.get("similar_cases_narrative") or "")[:400],
                            "price_insight_preview": (out.get("price_insight_narrative") or "")[:400],
                            "batch_count": len(out.get("batch_results") or []) if mode == "multi" else 0,
                        }
                    )

                br = out.get("batch_results") or []
                if br:
                    with st.expander("공시별로 보기", expanded=False):
                        for b in br:
                            lv_raw = str(b.get("impact_level") or "LOW").upper()
                            lv_ko = {"HIGH": "높음", "MEDIUM": "보통", "LOW": "낮음"}.get(lv_raw, lv_raw)
                            sk = "**건너뜀** " if b.get("skipped") else ""
                            st.markdown(
                                f"**{sk}{b.get('disclosure_title', '')[:80]}** — 영향 {lv_ko} · {b.get('disclosure_type', '')}"
                            )
                            if not b.get("skipped") and (b.get("similar_cases_narrative") or "").strip():
                                st.caption(
                                    (b.get("similar_cases_narrative") or "")[:500]
                                    + ("…" if len(str(b.get("similar_cases_narrative") or "")) > 500 else "")
                                )

                with st.expander("원본 데이터 (점검용)", expanded=False):
                    st.json({k: v for k, v in out.items() if k not in ("raw_body", "batch_results")})
                    rb = out.get("raw_body") or ""
                    if rb:
                        st.text_area("본문 일부", rb[:8000], height=240)

    with tab2:
        st.markdown("숫자·공시·재무를 물어보면 찾아서 답해 줍니다. **회사 이름**이나 **공시용 8자리 번호**를 알려 주세요.")
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "chat_pending" not in st.session_state:
            st.session_state.chat_pending = False

        for m in st.session_state.chat_messages:
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                continue
            with st.chat_message(role):
                c = m.content
                st.markdown(c if isinstance(c, str) else str(c))

        q = st.chat_input("질문을 입력하세요…")
        if q:
            st.session_state.chat_messages.append(HumanMessage(content=q))
            st.session_state.chat_pending = True

        if st.session_state.chat_pending:
            st.session_state.chat_pending = False
            with st.spinner("답변 준비 중…"):
                try:
                    agent = get_finance_react()
                    result = agent.invoke(
                        {"messages": st.session_state.chat_messages},
                        config={"recursion_limit": 40},
                    )
                    st.session_state.chat_messages = result.get("messages", [])
                except Exception as e:
                    st.session_state.chat_messages.pop()
                    st.error(str(e))
            st.rerun()


if __name__ == "__main__":
    main()
