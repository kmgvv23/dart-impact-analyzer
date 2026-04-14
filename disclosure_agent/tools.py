"""DART / KRX / 웹 검색 도구 — LangChain @tool."""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from datetime import datetime
from typing import Any, Optional
from xml.etree import ElementTree

import requests
from langchain_core.tools import tool

from disclosure_agent.config import get_dart_api_key
from disclosure_agent.krx_openapi import forward_returns_from_event_krx_api, latest_quote_krx


_DART_BASE = "https://opendart.fss.or.kr/api"


def _dart_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    key = get_dart_api_key()
    if not key:
        return {"error": "DART_API_KEY 미설정", "status": "no_key"}
    p = {"crtfc_key": key, **params}
    r = requests.get(f"{_DART_BASE}/{path}", params=p, timeout=60)
    r.raise_for_status()
    if path.endswith(".json"):
        return r.json()
    return {"raw": r.text}


def _find_document_element(root: ElementTree.Element) -> Optional[ElementTree.Element]:
    doc = root.find("document")
    if doc is not None:
        return doc
    for el in root.iter():
        if el.tag == "document" or el.tag.endswith("}document"):
            return el
    return None


def _zip_first_member_text(raw_zip: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = zf.namelist()
        if not names:
            return ""
        data = zf.read(names[0])
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _document_b64_from_regex(text: str) -> Optional[str]:
    m = re.search(r"<document[^>]*>([\s\S]*?)</document>", text, re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()
    inner = re.sub(r"^<!\[CDATA\[", "", inner)
    inner = re.sub(r"\]\]>$", "", inner)
    return inner.strip() or None


def _decode_document_zip_xml(content: bytes) -> str:
    """DART document.xml: (1) ZIP 직접 반환 (2) JSON 오류 (3) XML 래퍼+base64 ZIP."""
    # 최신 OPENDART: 응답 본문이 ZIP 바이너리인 경우가 많음 (Content-Type: application/x-msdownload)
    if len(content) >= 4 and content[:2] == b"PK":
        return _zip_first_member_text(content)

    text: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = content.decode("utf-8", errors="replace")
    text = text.replace("\x00", "")
    head = text.lstrip()[:400].lstrip("\ufeff")
    if head.startswith("{") or head.startswith("["):
        try:
            j = json.loads(text)
            raise ValueError(
                f"DART document API: status={j.get('status', '')} message={j.get('message', j)}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"DART document 응답이 JSON/XML이 아닙니다: {e}") from e

    root: Optional[ElementTree.Element] = None
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        b64 = _document_b64_from_regex(text)
        if b64:
            try:
                raw_zip = base64.b64decode(re.sub(r"\s+", "", b64))
                return _zip_first_member_text(raw_zip)
            except Exception as e:
                raise ValueError(f"DART 첨부 ZIP 디코딩 실패: {e}") from e
        raise

    doc_el = _find_document_element(root)
    if doc_el is None or not (doc_el.text and doc_el.text.strip()):
        status = root.findtext("status", "") or root.findtext(".//status", "")
        msg = root.findtext("message", "") or root.findtext(".//message", "")
        raise ValueError(f"DART document 응답 이상: status={status} message={msg}")

    b64 = doc_el.text.strip()
    try:
        raw_zip = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise ValueError(f"DART document base64 디코딩 실패: {e}") from e
    return _zip_first_member_text(raw_zip)


@tool
def dart_disclosure_fetch(rcept_no: str) -> str:
    """DART 공시 접수번호(rcept_no, 14자리)로 공시 원문(또는 첫 첨부 XML) 텍스트를 가져온다."""
    rcept_no = rcept_no.strip()
    if len(rcept_no) != 14 or not rcept_no.isdigit():
        return json.dumps({"ok": False, "error": "rcept_no는 14자리 숫자여야 합니다."}, ensure_ascii=False)
    key = get_dart_api_key()
    if not key:
        return json.dumps(
            {"ok": False, "error": "DART_API_KEY가 없어 원문을 가져올 수 없습니다. .env를 확인하세요."},
            ensure_ascii=False,
        )
    try:
        r = requests.get(
            f"{_DART_BASE}/document.xml",
            params={"crtfc_key": key, "rcept_no": rcept_no},
            timeout=90,
        )
        r.raise_for_status()
        text = _decode_document_zip_xml(r.content)
        if len(text) > 120_000:
            text = text[:120_000] + "\n...[truncated]"
        return json.dumps({"ok": True, "chars": len(text), "body": text}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def dart_disclosure_search(
    corp_code: str,
    disclosure_type_hint: str,
    bgn_de: str,
    end_de: str,
    max_rows: int = 30,
) -> str:
    """동일 법인(corp_code)에서 기간·유형 힌트로 공시 목록을 검색한다. 완화 루프에 사용."""
    corp_code = corp_code.strip()
    max_rows = max(1, min(int(max_rows), 100))
    key = get_dart_api_key()
    if not key:
        return json.dumps({"list": [], "error": "DART_API_KEY 미설정"}, ensure_ascii=False)
    try:
        data = _dart_get(
            "list.json",
            {"corp_code": corp_code, "bgn_de": bgn_de, "end_de": end_de, "page_count": max_rows, "page_no": 1},
        )
        if data.get("status") != "000":
            return json.dumps({"list": [], "dart_message": data.get("message")}, ensure_ascii=False)
        lst = data.get("list") or []
        hint = disclosure_type_hint.lower()
        filtered = []
        for row in lst:
            nm = (row.get("report_nm") or "") + " " + (row.get("rm") or "")
            if not hint or hint in nm.lower():
                filtered.append(
                    {
                        "rcept_no": row.get("rcept_no"),
                        "report_nm": row.get("report_nm"),
                        "rcept_dt": row.get("rcept_dt"),
                        "corp_cls": row.get("corp_cls"),
                    }
                )
        return json.dumps({"list": filtered[:max_rows]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"list": [], "error": str(e)}, ensure_ascii=False)


@tool
def dart_financials(corp_code: str, year: str, reprt_code: str = "11011") -> str:
    """
    DART 단일회계 항목 연결재무제표 요약.
    reprt_code: 11011(사업보고서), 11012(반기), 11013(1분기), 11014(3분기)
    """
    key = get_dart_api_key()
    if not key:
        return json.dumps({"error": "DART_API_KEY 미설정"}, ensure_ascii=False)
    try:
        data = _dart_get(
            "fnlttSinglAcntAll.json",
            {
                "corp_code": corp_code,
                "bsns_year": year,
                "reprt_code": reprt_code,
                "fs_div": "CFS",
            },
        )
        if data.get("status") != "000":
            return json.dumps({"error": data.get("message"), "rows": []}, ensure_ascii=False)
        rows = data.get("list") or []
        slim = [
            {
                "sj_nm": x.get("sj_nm"),
                "account_nm": x.get("account_nm"),
                "thstrm_amount": x.get("thstrm_amount"),
            }
            for x in rows[:40]
        ]
        return json.dumps({"rows": slim, "count": len(rows)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "rows": []}, ensure_ascii=False)


def _parse_yyyymmdd(s: str) -> Optional[datetime]:
    s = re.sub(r"\D", "", s)
    if len(s) >= 8:
        try:
            return datetime.strptime(s[:8], "%Y%m%d")
        except ValueError:
            return None
    return None


def _krx_out_needs_yfinance_fallback(out: dict) -> bool:
    if out.get("error"):
        return True
    if out.get("base_close") is None:
        return True
    fwd = out.get("forward_returns_pct") or out.get("returns_pct") or {}
    if not isinstance(fwd, dict) or not fwd:
        return True
    return False


@tool
def krx_price(stock_code: str, event_date_yyyymmdd: str) -> str:
    """
    KRX OPEN API(일별매매정보)로 공시일(또는 이후 첫 거래일) 종가 대비
    1W/1M/3M/6M/1Y(거래일 오프셋) 전방 수익률(%)을 계산한다.
    KRX가 비어 있거나(403·권한·망 분리 등) 실패하면 Yahoo Finance 보조 시세로 동일 지표를 채운다.
    .env: KRX_OPENAPI_KEY, 선택 KRX_OPENAPI_BASE, KRX_OPENAPI_USE_SAMPLE
    """
    from disclosure_agent.price_yfinance import forward_returns_from_event_yfinance, latest_quote_yfinance

    code = re.sub(r"\D", "", stock_code)
    if len(code) != 6:
        return json.dumps({"error": "stock_code는 6자리 숫자여야 합니다."}, ensure_ascii=False)
    if not _parse_yyyymmdd(event_date_yyyymmdd):
        return json.dumps({"error": "event_date_yyyymmdd 파싱 실패"}, ensure_ascii=False)
    try:
        out = forward_returns_from_event_krx_api(code, event_date_yyyymmdd)
        if not isinstance(out, dict):
            out = {"error": "KRX 응답 형식 오류"}
        if _krx_out_needs_yfinance_fallback(out):
            fb = forward_returns_from_event_yfinance(code, event_date_yyyymmdd)
            if isinstance(fb, dict) and not fb.get("error"):
                return json.dumps(fb, ensure_ascii=False)
            if isinstance(fb, dict) and fb.get("error") and out.get("error"):
                out["yahoo_fallback_error"] = fb["error"]
        if isinstance(out, dict):
            out.setdefault("latest_quote", latest_quote_krx(code))
            lq = out.get("latest_quote") if isinstance(out.get("latest_quote"), dict) else {}
            if not lq.get("ok"):
                ylq = latest_quote_yfinance(code)
                if ylq.get("ok"):
                    out["latest_quote"] = ylq
                    out.setdefault(
                        "price_source_note",
                        "최근 종가는 Yahoo Finance 보조 시세입니다.",
                    )
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def web_search(query: str) -> str:
    """짧은 웹 검색(컨센서스·맥락). 실패 시 빈 요약."""
    q = query.strip()
    if not q:
        return json.dumps({"summary": "", "results": []}, ensure_ascii=False)
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=5):
                results.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
        return json.dumps({"summary": f"{len(results)}건", "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"summary": "", "error": str(e), "results": []}, ensure_ascii=False)


DISCLOSURE_TOOLS = [dart_disclosure_fetch, dart_disclosure_search, krx_price, web_search]
CHAT_TOOLS = [dart_financials, dart_disclosure_search, web_search]
