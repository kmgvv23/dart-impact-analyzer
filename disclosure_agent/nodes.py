"""StateGraph 노드 — Supervisor 분기 전제."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from disclosure_agent.config import get_dart_api_key, get_openai_api_key
from disclosure_agent.dart_corp_codes import get_corp_rows, resolve_corp_by_name
from disclosure_agent.rules import (
    classify_disclosure_type,
    collapse_title_ws,
    extract_rcept_no,
    is_grey_zone,
)
from disclosure_agent.spec_digest import spec_block_for
from disclosure_agent.state import AgentState
from disclosure_agent.tools import dart_disclosure_fetch, krx_price, web_search


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


# 사용자 화면: 한글 숫자(십구만…) 방지
_REPORT_NUMERIC_ARABIC = (
    "표기(필수): 금액·비율·퍼센트·%p·배수·거래일·연·월·일·원·조·억·만 등 모든 수치는 아라비아 숫자(0-9)만 사용한다. "
    "한글로 읽는 숫자 표현(예: 십구만육천원, 삼십이퍼센트, 백삼십삼조원)은 절대 쓰지 않는다. "
    "날짜는 '2025-04-07' 또는 '2025년 4월 7일'처럼 숫자가 드러나게만 쓴다."
)
# 시나리오 소제목은 Streamlit에서 Base/Best/Worst로 붙임
_REPORT_SCENARIO_BODY_RULE = (
    "scenario_base·scenario_best·scenario_worst 각 필드 본문에는 '베이스''베스트''워스트' 등 시나리오 제목 문구를 넣지 말고 내용만 서술한다."
)


def _years_ago_yyyymmdd(years: int) -> str:
    return (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")


def _parse_input(text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "", ""
    first = lines[0]
    hint = lines[1] if len(lines) > 1 else ""
    return first, hint


def _company_lookup(corp_name: str) -> tuple[str, str, str]:
    """(corp_code, corp_name_official, stock_code) 또는 실패 시 ('','','')."""
    if not corp_name.strip():
        return "", "", ""
    if not get_dart_api_key():
        return "", "", ""
    hit = resolve_corp_by_name(corp_name.strip())
    if not hit:
        return "", "", ""
    return (
        str(hit.get("corp_code") or ""),
        str(hit.get("corp_name") or ""),
        str(hit.get("stock_code") or "").zfill(6),
    )


def _dart_list_max_pages() -> int:
    try:
        return max(1, min(20, int(os.getenv("DART_LIST_MAX_PAGES", "10"))))
    except ValueError:
        return 10


def _dart_scope_max_pages() -> int:
    try:
        return max(1, min(10, int(os.getenv("DART_LIST_SCOPE_MAX_PAGES", "4"))))
    except ValueError:
        return 4


def _list_scope_mode() -> str:
    return (os.getenv("DART_LIST_SCOPE", "all") or "all").strip().lower()


def _dedupe_rows_latest_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_r: dict[str, dict[str, Any]] = {}
    for x in rows:
        rno = str(x.get("rcept_no") or "")
        if len(rno) != 14:
            continue
        if rno not in by_r:
            by_r[rno] = x
    return sorted(by_r.values(), key=lambda z: str(z.get("rcept_dt", "")), reverse=True)


def _merge_row_lists(*parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for p in parts:
        merged.extend(p)
    return _dedupe_rows_latest_first(merged)


def _fetch_disclosure_list(
    corp_code: str,
    bgn_de: str,
    end_de: str,
    page_count: int = 100,
    *,
    pblntf_ty: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> list[dict[str, Any]]:
    """DART list.json — page_count 최대 100, 메가캡은 page_no 루프로 확장."""
    key = get_dart_api_key()
    if not key or not corp_code:
        return []
    import requests

    pc = min(int(page_count), 100)
    mp = max_pages if max_pages is not None else _dart_list_max_pages()
    acc: list[dict[str, Any]] = []
    for page_no in range(1, mp + 1):
        params: dict[str, Any] = {
            "crtfc_key": key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page_no,
            "page_count": pc,
            "sort": "date",
            "sort_mth": "desc",
        }
        if pblntf_ty:
            params["pblntf_ty"] = str(pblntf_ty).strip().upper()[:1]
        r = requests.get("https://opendart.fss.or.kr/api/list.json", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "000":
            break
        chunk = data.get("list") or []
        if not chunk:
            break
        acc.extend(chunk)
        if len(chunk) < pc:
            break
    return _dedupe_rows_latest_first(acc)


def _fetch_rows_for_narrow_scope(corp_code: str, bgn_de: str, end_de: str) -> list[dict[str, Any]]:
    """정기공시(A) 제외에 가깝게: 주요·발행·지분·거래소·기타만 별도 조회 후 병합."""
    raw = os.getenv("DART_LIST_PBLNTF_TY", "B,C,D,I,E")
    types: list[str] = []
    for x in raw.split(","):
        s = (x or "").strip().upper()
        if s and s[0].isalpha():
            types.append(s[0])
    if not types:
        types = ["B", "C", "D", "I", "E"]
    sp = _dart_scope_max_pages()
    parts = [_fetch_disclosure_list(corp_code, bgn_de, end_de, 100, pblntf_ty=ty, max_pages=sp) for ty in types]
    return _merge_row_lists(*parts)


def _score_disclosure_title(report_nm: str) -> int:
    """자동 선택 시: 시장 이슈 공시 우선, 정기·정정·저관여는 후순위."""
    nm = collapse_title_ws(report_nm or "")
    if any(k in nm for k in ("정정", "기재정정", "정정신고")):
        return -100
    high = (
        "유상증자",
        "유상 증자",
        "전환사채",
        "신주인수권부사채",
        "잠정실적",
        "연결실적",
        "분기실적",
        "반기실적",
        "합병",
        "분할",
        "최대주주",
        "대주주",
        "대규모계약",
        "단일판매",
        "공급계약",
        "관리종목",
        "상장폐지",
        "상장적격성",
        "CB",
        "BW",
        "유증",
    )
    low = (
        "사업보고",
        "반기보고",
        "분기보고",
        "감사보고서",
        "연결감사",
        "내부회계",
        "주주총회",
        "주총",
        "결산",
        "월간",
        "소각",
        "자기주식",
        "채무증권",
        "기업설명회",
        "공정공시",
        "주식등의",
        "대량보유",
    )
    s = 0
    for k in high:
        if k in nm:
            s += 25
    for k in low:
        if k in nm:
            s -= 8
    # 메가캡 다공시: '최대주주' 등만으로 +25가 여러 번 쌓여 잠정실적이 상위 N에서 밀리는 것 방지
    if classify_disclosure_type(nm, "") == "잠정실적":
        s += 180
    return s


def _newest_earnings_disclosure_row(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """접수일 최신 순 목록에서, 제목만으로 잠정실적(룰)에 해당하는 첫 공시."""
    for x in rows:
        tit = str(x.get("report_nm") or "")
        if is_grey_zone(tit, ""):
            continue
        if classify_disclosure_type(tit, "") == "잠정실적":
            return x
    return None


def _multi_max() -> int:
    try:
        return max(1, min(30, int(os.getenv("MULTI_DISCLOSURE_MAX", "5"))))
    except ValueError:
        return 5


def _build_company_and_queue(name: str) -> dict[str, Any]:
    """법인명만 입력 시: 최근 공시 목록에서 점수·일자 기준 상위 N건 큐."""
    name = (name or "").strip()
    out: dict[str, Any] = {"error": "", "pick_reason": "", "queue": []}
    if not name:
        out["error"] = "회사명(법인명)을 입력해 주세요."
        return out
    if not get_dart_api_key():
        out["error"] = "DART_API_KEY가 .env에 없습니다."
        return out
    corp_code, corp_name, stock_code = _company_lookup(name)
    if not corp_code:
        rows = get_corp_rows()
        if not rows:
            out["error"] = (
                "DART 고유번호 목록(corpCode.xml)을 받지 못했습니다. "
                "DART_API_KEY·네트워크를 확인하거나 잠시 후 다시 시도하세요."
            )
        else:
            out["error"] = f"고유번호 목록에서 회사를 찾지 못했습니다: {name}"
        return out
    bgn = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
    end = _today_yyyymmdd()
    scope = _list_scope_mode()
    if scope in ("narrow", "material"):
        rows = _fetch_rows_for_narrow_scope(corp_code, bgn, end)
    else:
        rows = _fetch_disclosure_list(corp_code, bgn, end, 100, max_pages=_dart_list_max_pages())
        if _newest_earnings_disclosure_row(rows) is None:
            extra = _fetch_disclosure_list(corp_code, bgn, end, 100, pblntf_ty="I", max_pages=6)
            rows = _merge_row_lists(rows, extra)
    if not rows:
        out["error"] = f"최근 공시 목록이 비어 있습니다: {corp_name}"
        return out
    scored: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for x in rows:
        rno = str(x.get("rcept_no") or "")
        if len(rno) != 14 or rno in seen:
            continue
        seen.add(rno)
        tit = str(x.get("report_nm") or "")
        sc = _score_disclosure_title(tit)
        scored.append((sc, str(x.get("rcept_dt") or ""), x))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    max_n = _multi_max()
    queue: list[dict[str, Any]] = []
    pinned = _newest_earnings_disclosure_row(rows)
    used_rcept: set[str] = set()
    if pinned is not None:
        pr = str(pinned.get("rcept_no") or "")
        if len(pr) == 14:
            used_rcept.add(pr)
            queue.append(
                {
                    "rcept_no": pr,
                    "report_nm": str(pinned.get("report_nm") or ""),
                    "rcept_dt": str(pinned.get("rcept_dt") or ""),
                    "score": _score_disclosure_title(str(pinned.get("report_nm") or "")),
                    "pinned": True,
                }
            )
    for sc, _rdt, x in scored:
        if len(queue) >= max_n:
            break
        rno = str(x.get("rcept_no") or "")
        if rno in used_rcept:
            continue
        used_rcept.add(rno)
        tit = str(x.get("report_nm") or "")
        queue.append({"rcept_no": rno, "report_nm": tit, "rcept_dt": str(x.get("rcept_dt") or ""), "score": sc})
    if not queue:
        out["error"] = "분석할 공시를 선택하지 못했습니다."
        return out
    out["corp_code"] = corp_code
    out["corp_name"] = corp_name
    out["stock_code"] = str(stock_code or "").zfill(6)
    out["queue"] = queue
    pin_note = " 최근 실적 속보가 있으면 맨 앞에 두었습니다." if pinned else ""
    out["pick_reason"] = (
        f"최근 공시 중 중요해 보이는 순으로 {len(queue)}건을 골랐습니다." + pin_note
    )
    return out


def _list_meta_for_rcept(corp_code: str, rcept_no: str) -> dict[str, Any]:
    key = get_dart_api_key()
    if not key or not corp_code:
        return {}
    import requests

    bgn = _years_ago_yyyymmdd(5)
    end = _today_yyyymmdd()
    r = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={
            "crtfc_key": key,
            "corp_code": corp_code,
            "bgn_de": bgn,
            "end_de": end,
            "page_no": 1,
            "page_count": 100,
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "000":
        return {}
    for row in data.get("list") or []:
        if str(row.get("rcept_no")) == rcept_no:
            return {
                "report_nm": row.get("report_nm"),
                "rcept_dt": row.get("rcept_dt"),
                "stock_code": str(row.get("stock_code", "") or "").zfill(6),
                "corp_name": row.get("corp_name"),
            }
    return {}


def ingest_node(state: AgentState) -> dict[str, Any]:
    raw = state.get("disclosure_input") or ""
    first, hint = _parse_input(raw)
    rcept = extract_rcept_no(raw) or extract_rcept_no(first)
    # 접수번호·URL이 있으면 둘째 줄만 법인 힌트(없으면 빈값). 없으면 첫 줄 = 회사명 검색.
    if rcept:
        corp_for_search = hint.strip()
    else:
        # 회사명 자동 검색: 첫 줄만 사용(둘째 줄 메모는 검색에 쓰지 않음)
        corp_for_search = (first or raw.strip()).strip()
    return {
        "corp_name_hint": corp_for_search,
        "rcept_no": rcept or "",
        "disclosure_input": raw,
    }


def load_dart_node(state: AgentState) -> dict[str, Any]:
    rcept = (state.get("rcept_no") or "").strip()
    pick_reason = ""
    corp_code = (state.get("corp_code") or "").strip()
    corp_name = (state.get("corp_name") or "").strip()
    stock_code = (state.get("stock_code") or "").strip()
    title = ""
    rcept_dt = ""

    if len(rcept) != 14:
        built = _build_company_and_queue(state.get("corp_name_hint") or "")
        if built.get("error"):
            return {
                "early_exit_reason": built["error"],
                "impact_level": "LOW",
                "raw_body": "",
                "disclosure_title": "",
                "disclosure_pick_reason": "",
                "analysis_mode": "single",
                "disclosure_queue": [],
                "batch_results": [],
            }
        q = built.get("queue") or []
        pick_reason = str(built.get("pick_reason") or "")
        corp_code = str(built.get("corp_code") or "")
        corp_name = str(built.get("corp_name") or "")
        stock_code = str(built.get("stock_code") or "").zfill(6)
        return {
            "analysis_mode": "multi",
            "disclosure_queue": q,
            "disclosure_pick_reason": pick_reason,
            "corp_code": corp_code,
            "corp_name": corp_name,
            "stock_code": stock_code,
            "raw_body": "",
            "rcept_no": "",
            "disclosure_title": f"다중 공시 분석 ({len(q)}건)",
            "disclosure_date": "",
            "early_exit_reason": "",
            "batch_results": [],
        }

    raw = dart_disclosure_fetch.invoke({"rcept_no": rcept})
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "early_exit_reason": "공시 원문 파싱 실패",
            "impact_level": "LOW",
            "raw_body": "",
            "disclosure_title": "",
            "disclosure_pick_reason": pick_reason,
            "analysis_mode": "single",
            "disclosure_queue": [],
            "batch_results": [],
        }
    if not payload.get("ok"):
        return {
            "early_exit_reason": payload.get("error", "DART 조회 실패"),
            "impact_level": "LOW",
            "raw_body": "",
            "disclosure_title": "",
            "disclosure_pick_reason": pick_reason,
            "analysis_mode": "single",
            "disclosure_queue": [],
            "batch_results": [],
        }
    body = payload.get("body") or ""
    hint = state.get("corp_name_hint") or ""
    if not corp_code:
        corp_code, corp_name, stock_code = _company_lookup(hint)
    meta = _list_meta_for_rcept(corp_code, rcept) if corp_code else {}
    if meta.get("report_nm"):
        title = str(meta.get("report_nm") or title)
    if meta.get("rcept_dt"):
        rcept_dt = str(meta.get("rcept_dt") or rcept_dt)
    sc = str(meta.get("stock_code") or "").zfill(6) if meta.get("stock_code") else stock_code
    if meta.get("corp_name"):
        corp_name = str(meta.get("corp_name"))
    if not title:
        title = body[:120].replace("\n", " ")
    return {
        "analysis_mode": "single",
        "disclosure_queue": [],
        "batch_results": [],
        "rcept_no": rcept,
        "raw_body": body,
        "disclosure_title": title,
        "disclosure_date": rcept_dt,
        "corp_code": corp_code,
        "corp_name": corp_name,
        "stock_code": sc,
        "early_exit_reason": "",
        "disclosure_pick_reason": pick_reason,
    }


def _gather_consensus_snippet(corp: str, title: str, dtype: str) -> str:
    """잠정실적: 컨센·서프라이즈 맥락 1회 웹 검색(지연 최소화)."""
    corp = (corp or "").strip()
    if dtype != "잠정실적" or not corp:
        return ""
    tit = (title or "").strip()[:40]
    q = f"{corp} {tit} 잠정실적 컨센서스 영업이익 추정 서프라이즈"
    try:
        return str(web_search.invoke({"query": q}))[:3500]
    except Exception:
        return ""


def classify_rules_node(state: AgentState) -> dict[str, Any]:
    title = state.get("disclosure_title") or ""
    body = state.get("raw_body") or ""
    grey = is_grey_zone(title, body)
    dtype = None if grey else classify_disclosure_type(title, body)
    return {
        "grey_zone": grey,
        "disclosure_type": dtype or "비대상",
        "early_exit_reason": "",
    }


class Judgement(BaseModel):
    """OpenAI response_format은 object 타입에 additionalProperties:false가 필요해 dict 필드 불가."""

    model_config = ConfigDict(extra="forbid")

    extracted_fields_json: str = Field(
        default="{}",
        description='유형별 추출 수치·문구를 하나의 JSON 객체로 직렬화한 문자열. 예: {"희석률":"12%","목적":"운영자금"}',
    )
    llm_stance: str = ""
    impact_level: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    consensus_note: str = ""


def _llm_judge_run(
    dtype: str,
    title: str,
    body: str,
    corp: str,
    consensus_snippet: str,
) -> dict[str, Any]:
    """룰 분류 이후 LLM 판정 (단일·배치 공통)."""
    body = (body or "")[:20_000]
    spec = spec_block_for(dtype)
    earn_extra = ""
    if dtype == "잠정실적":
        earn_extra = (
            "잠정실적: 컨센서스_웹보조와 공시 수치를 대조하고, consensus_gap_pct·yoy_pct·one_off_items 등을 "
            "가능하면 extracted_fields_json에 숫자 또는 문구로 넣어라. "
            "컨센 대비 유의미한 우열이 수치로 설명되면 HIGH를 검토하고, 웹 근거가 빈약·상충이면 MEDIUM 이하로 보수적으로 분류하라. "
        )
    system = (
        spec
        + "\n당신은 한국 상장사 공시 분석가입니다. 출력은 JSON 스키마에 맞춘다. "
        + earn_extra
        + "잠정실적에서 컨센서스 수치가 웹 요약에서 확실치 않으면 consensus_values 키를 '미확인'으로 둔다. "
        "extracted_fields_json은 반드시 유효한 JSON 객체 하나의 문자열이다."
    )
    human = (
        f"법인명: {corp}\n"
        f"공시유형(룰): {dtype}\n"
        f"제목: {title}\n"
        f"컨센서스_웹보조:\n{consensus_snippet}\n"
        f"본문 일부:\n{body}\n"
    )
    model = ChatOpenAI(
        model="gpt-5-mini",
        api_key=get_openai_api_key(),
    )
    structured = model.with_structured_output(Judgement, strict=False)
    out: Judgement = structured.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ]
    )
    raw_ef = (out.extracted_fields_json or "").strip() or "{}"
    try:
        parsed = json.loads(raw_ef)
        merged = dict(parsed) if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        merged = {}
    if dtype == "잠정실적" and "consensus_values" not in merged:
        merged["consensus_values"] = "미확인"
    return {
        "extracted_fields": merged,
        "llm_stance": out.llm_stance,
        "impact_level": out.impact_level,
    }


def llm_judge_node(state: AgentState) -> dict[str, Any]:
    """유형별 명세·영향도. 잠정실적은 웹 검색으로 컨센서스 보조."""
    dtype = state.get("disclosure_type") or ""
    title = state.get("disclosure_title") or ""
    body = (state.get("raw_body") or "")[:24_000]
    corp = state.get("corp_name") or ""
    consensus = _gather_consensus_snippet(corp, title, dtype)
    return _llm_judge_run(dtype, title, body, corp, consensus)


class SimilarCasesOut(BaseModel):
    """MEDIUM·HIGH: 동일 법인 과거 공시 목록 없이 유형·섹터 참고 사례를 서술."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(
        default="",
        description="국내외 유사 섹터·유사 공시 유형의 전형적 사례와 시장 반응 패턴. 2~5문단 한국어. 단정·내부정보 금지.",
    )


def similar_cases_llm_node(state: AgentState) -> dict[str, Any]:
    """동일 법인 DART 목록 peer 제거: LLM으로 타사·유사 사례 맥락만 생성."""
    payload = {
        "corp_name": state.get("corp_name"),
        "stock_code": state.get("stock_code"),
        "disclosure_type": state.get("disclosure_type"),
        "title": state.get("disclosure_title"),
        "stance": state.get("llm_stance"),
        "extracted": state.get("extracted_fields"),
    }
    system = (
        "한국 상장사 공시 맥락 교육용 요약이다. 투자 권유·내부정보·확정적 재무예측은 금지. "
        "동일 회사의 DART 과거 공시 목록은 사용하지 않는다. "
        "아래 user payload에 대해 국내외 유사 섹터에서 '이 유형의 공시가 나왔을 때 시장이 흔히 참고하는 틀·유사 사례'를 "
        "일반론·공개된 역사적 유형 위주로 2~5문단 한국어로 쓴다. "
        "구체 기업명을 쓸 때는 널리 알려진 공개 사례에 한정하고, 불확실하면 '유사 사례에서 …'로 완곡하게. "
        "수치를 쓸 경우 아라비아 숫자만 쓴다(한글 숫자 금지)."
    )
    model = ChatOpenAI(model="gpt-5-mini", api_key=get_openai_api_key())
    structured = model.with_structured_output(SimilarCasesOut, strict=False)
    out: SimilarCasesOut = structured.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)[:12_000]},
        ]
    )
    return {
        "peer_set": [],
        "similar_cases_narrative": (out.narrative or "").strip(),
        "peer_search_note": "아래는 공개된 시장 관행·유형을 바탕으로 한 참고용 요약입니다. 특정 거래·기업의 사실을 보장하지 않습니다.",
    }


class PriceContextOut(BaseModel):
    """시세 수치 + 공시 맥락 주가 해설(리포트 peer_price_digest 원천)."""

    model_config = ConfigDict(extra="forbid")

    digest: str = Field(
        default="",
        description="2~5문단 한국어. 금액·날짜·등락은 payload 숫자와 일치, 아라비아 숫자로만 표기. 영문 변수명·JSON 키·API명 금지.",
    )


def price_context_llm_node(state: AgentState) -> dict[str, Any]:
    """현재가(최근 종가)·KRX 수익률 payload를 바탕으로 주가 맥락만 LLM 서술."""
    pp = state.get("price_pattern") or {}
    payload = {
        "corp_name": state.get("corp_name"),
        "stock_code": state.get("stock_code"),
        "disclosure_title": state.get("disclosure_title"),
        "disclosure_type": state.get("disclosure_type"),
        "disclosure_date": state.get("disclosure_date"),
        "impact_level": state.get("impact_level"),
        "llm_stance": state.get("llm_stance"),
        "krx_price_payload": pp,
        "similar_cases_excerpt": (state.get("similar_cases_narrative") or "")[:2500],
    }
    system = (
        "한국 주식 시세·공시 맥락 요약가. 투자 권유 금지. "
        "user JSON의 krx_price_payload에 있는 숫자·날짜·종목명만 인용(임의 수치 금지). "
        "출력 digest는 일반 투자자용 한국어 문장만: 영문 변수명·스네이크케이스·JSON 키·필드명·괄호 속 기술표기를 쓰지 마라. "
        + _REPORT_NUMERIC_ARABIC
        + " 가격·수익률 예: '2025-04-07 종가 196,500원', '이후 5거래일 누적 +4.8%'. "
        "시세 오류가 있으면 '시세를 불러오지 못했다' 수준으로만 짧게 알린다. "
        "similar_cases_excerpt는 배경 참고만이며 사실 주장으로 섞지 말 것. 2~5문단."
    )
    model = ChatOpenAI(model="gpt-5-mini", api_key=get_openai_api_key())
    structured = model.with_structured_output(PriceContextOut, strict=False)
    out: PriceContextOut = structured.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)[:14_000]},
        ]
    )
    return {"price_insight_narrative": (out.digest or "").strip()}


def price_pattern_node(state: AgentState, light: bool) -> dict[str, Any]:
    code = state.get("stock_code") or ""
    d = state.get("disclosure_date") or _today_yyyymmdd()
    if len(code) != 6:
        return {"price_pattern": {"note": "종목코드 없음", "returns_pct": {}}}
    raw = krx_price.invoke({"stock_code": code, "event_date_yyyymmdd": d})
    try:
        pat = json.loads(raw)
    except json.JSONDecodeError:
        pat = {"error": "parse"}
    if light:
        pat["mode"] = "light"
    else:
        pat["mode"] = "full"
    return {"price_pattern": pat}


class ReportOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description="핵심 요약 3~7문장. 수치·날짜는 아라비아 숫자만. 내부 필드명·영문 변수·코드식 괄호 금지."
    )
    impact_positive: str = Field(
        default="",
        description="유리·긍정 요인 2~6문장. 수치는 아라비아 숫자만. 접두 라벨 없이 본문만.",
    )
    impact_negative: str = Field(
        default="",
        description="불리·리스크·불확실 2~6문장. 수치는 아라비아 숫자만. 접두 라벨 없이 본문만.",
    )
    peer_price_digest: str = Field(
        default="",
        description="공시 전후 시세 맥락 2~6문장. 금액·날짜·%는 아라비아 숫자. 영문 변수·API명 금지.",
    )
    scenario_base: str = Field(
        default="",
        description="Base 시나리오 본문 2~5문장. 제목/소제목 없이. 수치는 아라비아 숫자만.",
    )
    scenario_best: str = Field(
        default="",
        description="Best 시나리오 본문 2~5문장. 제목/소제목 없이. 수치는 아라비아 숫자만.",
    )
    scenario_worst: str = Field(
        default="",
        description="Worst 시나리오 본문 2~5문장. 제목/소제목 없이. 수치는 아라비아 숫자만.",
    )
    action_stop_loss: str
    action_take_profit: str
    action_entry: str
    action_wait: str
    disclaimer: str


def report_writer_node(state: AgentState) -> dict[str, Any]:
    model = ChatOpenAI(model="gpt-5-mini", api_key=get_openai_api_key())
    structured = model.with_structured_output(ReportOut, strict=False)
    payload = {
        "disclosure_type": state.get("disclosure_type"),
        "impact_level": state.get("impact_level"),
        "title": state.get("disclosure_title"),
        "stance": state.get("llm_stance"),
        "extracted": state.get("extracted_fields"),
        "similar_cases_narrative": state.get("similar_cases_narrative", ""),
        "peer_search_note": state.get("peer_search_note", ""),
        "price_insight_narrative": state.get("price_insight_narrative", ""),
        "price": state.get("price_pattern"),
    }
    il = state.get("impact_level") or ""
    spec_note = ""
    if il == "MEDIUM":
        spec_note = (
            "영향도 MEDIUM: 서술 톤은 중립·보수적으로 유지한다. "
            "user의 참고 사례 텍스트·시세 해설·price 수치를 peer_price_digest에 균형 있게 반영한다. "
        )
    system = (
        "투자 판단 보조 리포트를 한국어로 작성합니다. "
        + spec_note
        + _REPORT_NUMERIC_ARABIC
        + " "
        + _REPORT_SCENARIO_BODY_RULE
        + " "
        "사용자에게 보이는 모든 필드(summary, impact_*, peer_price_digest, scenario_*, action_*, disclaimer)에 "
        "영문 변수명·스네이크케이스·JSON 키·API명·내부 필드명을 쓰거나 괄호로 언급하지 마라. "
        "peer_price_digest에는 user JSON의 시세 해설을 우선 반영해 다듬는다(수치·날짜는 user의 price와 일치, 아라비아 숫자). "
        "시세 해설이 비어 있으면 user의 price 시세·수익률만으로 짧게 작성한다. "
        "user의 참고 사례 텍스트는 배경일 뿐 팩트로 단정하지 말 것. "
        "impact_positive와 impact_negative는 서로 다른 문단으로 채운다(한쪽 비어도 됨). "
        "scenario_base·scenario_best·scenario_worst는 각각 별도 사건 전개만 쓴다. "
        "action_* 필드는 손절/익절/진입/관망 각 한두 문장. disclaimer는 한 번만 간결히."
    )
    out: ReportOut = structured.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)[:28_000]},
        ]
    )
    report = {
        "summary": out.summary,
        "impact_positive": out.impact_positive,
        "impact_negative": out.impact_negative,
        "peer_price_digest": out.peer_price_digest,
        "scenario_base": out.scenario_base,
        "scenario_best": out.scenario_best,
        "scenario_worst": out.scenario_worst,
        "actions": {
            "손절": out.action_stop_loss,
            "익절": out.action_take_profit,
            "진입": out.action_entry,
            "관망": out.action_wait,
        },
        "disclaimer": out.disclaimer,
    }
    return {"final_report": report}


def _rank_impact(lvl: str) -> int:
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get((lvl or "").upper(), 0)


def multi_batch_node(state: AgentState) -> dict[str, Any]:
    """회사명-only 경로: 큐의 각 공시를 fetch → 룰 → LLM → MEDIUM/HIGH면 유사 사례+주가."""
    queue = state.get("disclosure_queue") or []
    corp_code = state.get("corp_code") or ""
    corp_name = state.get("corp_name") or ""
    stock_code = (state.get("stock_code") or "").zfill(6)
    batch: list[dict[str, Any]] = []
    for item in queue:
        rcept = str(item.get("rcept_no") or "")
        title = str(item.get("report_nm") or "")
        rcept_dt = str(item.get("rcept_dt") or "")
        one: dict[str, Any] = {
            "rcept_no": rcept,
            "disclosure_title": title,
            "disclosure_date": rcept_dt,
            "skipped": False,
            "skip_reason": "",
            "grey_zone": False,
            "disclosure_type": "",
            "impact_level": "LOW",
            "llm_stance": "",
            "extracted_fields": {},
            "peer_set": [],
            "peer_search_note": "",
            "similar_cases_narrative": "",
            "price_insight_narrative": "",
            "price_pattern": {},
        }
        if len(rcept) != 14:
            one["skipped"] = True
            one["skip_reason"] = "접수번호 형식 오류"
            batch.append(one)
            continue
        raw = dart_disclosure_fetch.invoke({"rcept_no": rcept})
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            one["skipped"] = True
            one["skip_reason"] = "공시 원문 파싱 실패"
            batch.append(one)
            continue
        if not payload.get("ok"):
            one["skipped"] = True
            one["skip_reason"] = str(payload.get("error") or "DART 조회 실패")
            batch.append(one)
            continue
        body = str(payload.get("body") or "")
        sub: AgentState = {
            "raw_body": body,
            "disclosure_title": title,
            "disclosure_date": rcept_dt,
            "corp_code": corp_code,
            "corp_name": corp_name,
            "stock_code": stock_code,
            "rcept_no": rcept,
        }
        cls_out = classify_rules_node(sub)
        dtype = str(cls_out.get("disclosure_type") or "")
        grey = bool(cls_out.get("grey_zone"))
        one["grey_zone"] = grey
        one["disclosure_type"] = dtype
        if grey:
            one["skipped"] = True
            one["skip_reason"] = "회색지대(정정 등) 제외"
            batch.append(one)
            continue
        if dtype == "비대상":
            one["skipped"] = True
            one["skip_reason"] = "룰 기준 비대상 유형"
            batch.append(one)
            continue
        consensus = _gather_consensus_snippet(corp_name, title, dtype)
        jd = _llm_judge_run(dtype, title, body, corp_name, consensus)
        one.update(jd)
        impact = str(jd.get("impact_level") or "MEDIUM")
        ev_date = rcept_dt if len(str(rcept_dt)) == 8 else _today_yyyymmdd()
        pstate: AgentState = {**sub, "disclosure_type": dtype, **jd}
        if impact in ("HIGH", "MEDIUM"):
            sc_out = similar_cases_llm_node(pstate)
            one["peer_set"] = []
            one["similar_cases_narrative"] = str(sc_out.get("similar_cases_narrative") or "")
            one["peer_search_note"] = str(sc_out.get("peer_search_note") or "")
            px = price_pattern_node({**pstate, "disclosure_date": ev_date}, light=False)
            one["price_pattern"] = px.get("price_pattern") or {}
            pc_state: AgentState = {
                **pstate,
                **jd,
                "similar_cases_narrative": one["similar_cases_narrative"],
                "price_pattern": one["price_pattern"],
            }
            one["price_insight_narrative"] = str(price_context_llm_node(pc_state).get("price_insight_narrative") or "")
        batch.append(one)

    best: Optional[dict[str, Any]] = None
    best_r = 0
    for b in batch:
        if b.get("skipped"):
            continue
        r = _rank_impact(str(b.get("impact_level") or ""))
        if r >= best_r:
            best_r = r
            best = b
    primary = best or (batch[0] if batch else {})
    return {
        "batch_results": batch,
        "rcept_no": str(primary.get("rcept_no") or ""),
        "disclosure_title": state.get("disclosure_title") or "",
        "disclosure_date": str(primary.get("disclosure_date") or ""),
        "disclosure_type": str(primary.get("disclosure_type") or ""),
        "impact_level": str(primary.get("impact_level") or "LOW"),
        "grey_zone": bool(primary.get("grey_zone")),
        "extracted_fields": primary.get("extracted_fields") or {},
        "llm_stance": str(primary.get("llm_stance") or ""),
        "peer_set": [],
        "peer_search_note": str(primary.get("peer_search_note") or ""),
        "similar_cases_narrative": str(primary.get("similar_cases_narrative") or ""),
        "price_insight_narrative": str(primary.get("price_insight_narrative") or ""),
        "price_pattern": primary.get("price_pattern") or {},
        "early_exit_reason": "",
    }


def multi_report_writer_node(state: AgentState) -> dict[str, Any]:
    """배치 결과를 하나의 리포트로 요약."""
    model = ChatOpenAI(model="gpt-5-mini", api_key=get_openai_api_key())
    structured = model.with_structured_output(ReportOut, strict=False)
    slim: list[dict[str, Any]] = []
    for b in state.get("batch_results") or []:
        price = b.get("price_pattern") or {}
        slim.append(
            {
                "rcept_no": b.get("rcept_no"),
                "title": b.get("disclosure_title"),
                "date": b.get("disclosure_date"),
                "type": b.get("disclosure_type"),
                "impact": b.get("impact_level"),
                "stance": b.get("llm_stance"),
                "extracted": b.get("extracted_fields"),
                "skipped": b.get("skipped"),
                "skip_reason": b.get("skip_reason"),
                "similar_cases_narrative": (b.get("similar_cases_narrative") or "")[:4000],
                "peer_search_note": b.get("peer_search_note"),
                "price_insight_excerpt": (b.get("price_insight_narrative") or "")[:2000],
                "latest_quote": (price.get("latest_quote") if isinstance(price, dict) else None),
                "price_returns": price.get("returns_pct"),
                "price_mode": price.get("mode"),
            }
        )
    payload = {
        "corp_name": state.get("corp_name"),
        "stock_code": state.get("stock_code"),
        "items": slim,
    }
    system = (
        "여러 건의 공시 일괄 분석 결과를 종합해 투자 판단 보조 리포트를 한국어로 작성합니다. "
        + _REPORT_NUMERIC_ARABIC
        + " "
        + _REPORT_SCENARIO_BODY_RULE
        + " "
        "각 item의 참고 사례 문단은 팩트체크되지 않은 배경일 뿐이다. "
        "시세 해설 요약은 해당 건이 MEDIUM/HIGH일 때만 의미가 있다. "
        "HIGH·잠정실적 서프라이즈를 우선한다. 건과 건이 상충하면 명시한다. "
        "사용자에게 보이는 필드에 영문 변수명·JSON 키·내부 필드명을 쓰지 마라. "
        "impact_positive/impact_negative, scenario_base/best/worst는 각각 분리해 채운다. "
        "action_* 필드는 손절/익절/진입/관망 각각 한두 문장. disclaimer는 한 번만 간결히."
    )
    out: ReportOut = structured.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)[:28_000]},
        ]
    )
    report = {
        "summary": out.summary,
        "impact_positive": out.impact_positive,
        "impact_negative": out.impact_negative,
        "peer_price_digest": out.peer_price_digest,
        "scenario_base": out.scenario_base,
        "scenario_best": out.scenario_best,
        "scenario_worst": out.scenario_worst,
        "actions": {
            "손절": out.action_stop_loss,
            "익절": out.action_take_profit,
            "진입": out.action_entry,
            "관망": out.action_wait,
        },
        "disclaimer": out.disclaimer,
    }
    return {"final_report": report}


def early_exit_node(state: AgentState) -> dict[str, Any]:
    reason = state.get("early_exit_reason") or ""
    grey = state.get("grey_zone")
    dtype = state.get("disclosure_type")
    body = state.get("raw_body") or ""

    if reason and not body:
        return {
            "final_report": {
                "summary": reason,
                "impact_positive": "",
                "impact_negative": "",
                "peer_price_digest": "",
                "scenario_base": "",
                "scenario_best": "",
                "scenario_worst": "",
                "actions": {"손절": "-", "익절": "-", "진입": "-", "관망": "-"},
                "disclaimer": "본 서비스는 투자 권유가 아닌 정보 제공 목적입니다.",
            }
        }

    if grey:
        msg = (
            "정정·기재정정 등 회색지대 공시로 분류되어 본 분석 파이프라인에서 제외했습니다. "
            "정정 전후 대조 및 원 공시 확인을 권장합니다."
        )
    elif dtype == "비대상" or not body:
        msg = (
            "분석 대상 유형(유상증자, CB/BW, 잠정실적, 합병·분할, 최대주주변경, "
            "대규모계약, 관리종목·상장폐지)에 해당하지 않거나 정기 공시에 가깝습니다. "
            "핵심 변동 공시인지 제목·본문 키워드를 다시 확인해 보세요."
        )
    elif reason:
        msg = reason
    elif state.get("impact_level") == "LOW" and body:
        msg = (
            "영향도가 낮게 평가되어 유사사례·주가 심층 분석은 생략했습니다. "
            f"{state.get('llm_stance') or ''}"
        )
    else:
        msg = (
            "영향도가 낮은 것으로 판단되어 심층 유사사례·주가 단계는 생략했습니다. "
            "공시는 참고용이며 투자 책임은 투자자에게 있습니다."
        )

    stance = (state.get("llm_stance") or "").strip()
    summary = msg if not stance else f"{msg}\n\n{stance}"
    return {
        "final_report": {
            "summary": summary,
            "impact_positive": "",
            "impact_negative": "",
            "peer_price_digest": "",
            "scenario_base": "",
            "scenario_best": "",
            "scenario_worst": "",
            "actions": {"손절": "-", "익절": "-", "진입": "-", "관망": "-"},
            "disclaimer": "본 서비스는 투자 권유가 아닌 정보 제공 목적입니다.",
        }
    }


def route_after_load(state: AgentState) -> str:
    if state.get("early_exit_reason") and not (state.get("raw_body") or "").strip():
        return "early_exit"
    if state.get("analysis_mode") == "multi":
        return "multi_batch"
    return "classify_rules"


def route_after_classify(state: AgentState) -> str:
    if state.get("grey_zone"):
        return "early_exit"
    if (state.get("disclosure_type") or "") == "비대상":
        return "early_exit"
    return "llm_judge"


def route_after_judge(state: AgentState) -> str:
    if state.get("impact_level") == "LOW":
        return "early_exit"
    return "similar_cases"
