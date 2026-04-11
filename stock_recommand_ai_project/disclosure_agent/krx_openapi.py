"""
KRX 정보데이터시스템 OPEN API (data-dbg.krx.co.kr) — pykrx 미사용.

- 인증: HTTP 헤더 `AUTH_KEY` (openapi.krx.co.kr 발급 + 서비스 이용신청)
- 샘플: 키 미설정 시 공식 샘플 키·/svc/sample/apis/sto 경로 (제한 종목만)
- 일별매매정보: basDd(거래일)당 1회 호출 후 해당 종목 행만 사용
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Optional

import requests

# 공식 문서 샘플 인증키(공개) — 전체 시장 조회는 본인 키 필요
KRX_SAMPLE_AUTH_KEY = "74D1B99DFBF345BBA3FB4476510A4BED4C78D13A"

_KRX_JSON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; disclosure-agent/1.0)",
    "Accept": "application/json",
}


def _krx_api_base() -> str:
    return os.getenv("KRX_OPENAPI_BASE", "https://data-dbg.krx.co.kr").rstrip("/")


def _krx_auth_key() -> str:
    k = os.getenv("KRX_OPENAPI_KEY", "").strip()
    return k if k else KRX_SAMPLE_AUTH_KEY


def _krx_use_sample_path() -> bool:
    if os.getenv("KRX_OPENAPI_USE_SAMPLE", "").lower() in ("1", "true", "yes"):
        return True
    return not bool(os.getenv("KRX_OPENAPI_KEY", "").strip())


def _daily_trade_path(market_code: str) -> str:
    """finder marketCode: STK / KSQ / KNX"""
    m = (market_code or "STK").upper()
    if m == "KSQ":
        return "ksq_bydd_trd"
    if m == "KNX":
        return "knx_bydd_trd"
    return "stk_bydd_trd"


def _daily_trade_url(market_code: str) -> str:
    base = _krx_api_base()
    name = _daily_trade_path(market_code)
    prefix = "/svc/sample/apis/sto/" if _krx_use_sample_path() else "/svc/apis/sto/"
    return f"{base}{prefix}{name}"


def krx_finder_stock(stock_code: str) -> Optional[dict[str, Any]]:
    """단축코드 6자리 → full_code, marketCode (STK/KSQ/…)."""
    code = re.sub(r"\D", "", stock_code)
    if len(code) != 6:
        return None
    # finder는 정보데이터시스템 메인 호스트(로그인 불필요 구간)
    url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    data = {
        "bld": "dbms/comm/finder/finder_stkisu",
        "locale": "ko_KR",
        "mktsel": "ALL",
        "searchText": code,
        "typeNo": "0",
    }
    headers = {
        **_KRX_JSON_HEADERS,
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    try:
        r = requests.post(url, data=data, headers=headers, timeout=30)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None
    for row in js.get("block1") or []:
        if str(row.get("short_code", "")).zfill(6) == code:
            return {
                "full_code": row.get("full_code"),
                "short_code": str(row.get("short_code", "")).zfill(6),
                "codeName": row.get("codeName"),
                "marketCode": row.get("marketCode") or "STK",
            }
    return None


def _parse_close(row: dict[str, Any]) -> Optional[float]:
    raw = row.get("TDD_CLSPRC")
    if raw is None:
        return None
    s = str(raw).replace(",", "").strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_outblock_for_date(market_code: str, bas_dd: str) -> list[dict[str, Any]]:
    """일별매매정보 OutBlock_1. 401/403·HTML 차단 시 빈 리스트(호출부에서 보조 시세로 대체)."""
    url = _daily_trade_url(market_code)
    headers = {**_KRX_JSON_HEADERS, "AUTH_KEY": _krx_auth_key()}
    try:
        r = requests.get(url, params={"basDd": bas_dd}, headers=headers, timeout=45)
        if r.status_code in (401, 403):
            return []
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ct:
            return []
        r.raise_for_status()
        if "json" not in ct and not r.text.strip().startswith("{"):
            return []
        js = r.json()
    except Exception:
        return []
    if js.get("respCode") == "401" or "Unauthorized" in str(js.get("respMsg", "")):
        return []
    return js.get("OutBlock_1") or []


def build_forward_return_dict(
    closes: dict[str, float],
    event_date: date,
    stock_code_6: str,
    *,
    source: str,
    sample_mode: bool,
    latest_quote_override: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """공시일(또는 이후 첫 거래일) 종가 기준 전방 수익률 dict 조립 — KRX·보조 시세 공통."""
    if not closes:
        return {"error": "시세 맵이 비었습니다."}
    code6 = re.sub(r"\D", "", stock_code_6).zfill(6)
    ymd_sorted = sorted(closes.keys())
    ev_s = event_date.strftime("%Y%m%d")
    anchor_ymd: Optional[str] = None
    for y in ymd_sorted:
        if y >= ev_s:
            anchor_ymd = y
            break
    if anchor_ymd is None:
        anchor_ymd = ymd_sorted[-1]
    ordered = [y for y in ymd_sorted if y >= anchor_ymd]
    if not ordered:
        return {"error": "기준일 이후 데이터 없음"}
    p0 = float(closes[ordered[0]])
    windows = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    forward: dict[str, Any] = {}
    for label, step in windows.items():
        j = step
        if j < len(ordered):
            p1 = closes[ordered[j]]
            forward[label] = round((p1 / p0 - 1.0) * 100.0, 2)
        else:
            forward[label] = None
    lq = latest_quote_override if latest_quote_override is not None else latest_quote_krx(code6)
    out: dict[str, Any] = {
        "source": source,
        "sample_mode": sample_mode,
        "anchor_trading_day": anchor_ymd,
        "base_close": round(p0, 2),
        "forward_returns_pct": forward,
        "returns_pct": forward,
        "latest_quote": lq,
    }
    if extra:
        out.update(extra)
    return out


def _row_for_stock(rows: list[dict[str, Any]], stock_code: str) -> Optional[dict[str, Any]]:
    code = stock_code.zfill(6)
    for row in rows:
        isu = str(row.get("ISU_CD", "") or "").zfill(6)
        srt = str(row.get("ISU_SRT_CD", "") or "").zfill(6)
        if isu == code or srt == code:
            return row
    return None


def _row_for_stock_any_market(stock_code: str, bas_dd: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """finder 없이 STK→KSQ→KNX 순으로 단축코드 매칭."""
    code = re.sub(r"\D", "", stock_code).zfill(6)
    if len(code) != 6:
        return None, None
    for mkt in ("STK", "KSQ", "KNX"):
        rows = _fetch_outblock_for_date(mkt, bas_dd)
        row = _row_for_stock(rows, code)
        if row:
            return row, mkt
    return None, None


def latest_quote_krx(stock_code: str, max_calendar_days: int = 21) -> dict[str, Any]:
    """
    최근 영업일 기준 종가(현재가에 가장 가까운 공식 종가).
    finder 실패 시에도 일별매매정보에서 STK/KSQ/KNX 순 탐색.
    """
    code = re.sub(r"\D", "", stock_code).zfill(6)
    if len(code) != 6:
        return {"ok": False, "error": "stock_code는 6자리여야 합니다."}
    d = date.today()
    for _ in range(max_calendar_days):
        if d.weekday() < 5:
            bas_dd = d.strftime("%Y%m%d")
            row, mkt = _row_for_stock_any_market(code, bas_dd)
            if row:
                cl = _parse_close(row)
                if cl is not None:
                    return {
                        "ok": True,
                        "basDd": bas_dd,
                        "marketCode": mkt,
                        "close": cl,
                        "isu_nm": str(row.get("ISU_ABBRV") or row.get("ISU_NM") or ""),
                        "source": "KRX_OPENAPI",
                        "sample_mode": _krx_use_sample_path(),
                    }
        d -= timedelta(days=1)
    msg = "최근 거래일 시세를 찾지 못했습니다."
    if _krx_use_sample_path():
        msg += " 샘플 API는 일부 종목만 제공합니다. KRX_OPENAPI_KEY 설정을 권장합니다."
    return {"ok": False, "error": msg, "sample_mode": _krx_use_sample_path()}


def _daterange_weekdays(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _prefetch_closes(
    stock_code: str,
    market_code: str,
    dates: list[date],
    max_workers: int = 8,
) -> dict[str, float]:
    """basDd 병렬 조회 → YYYYMMDD -> 종가."""
    results: dict[str, float] = {}

    def job(d: date) -> tuple[str, Optional[float]]:
        ymd = d.strftime("%Y%m%d")
        rows = _fetch_outblock_for_date(market_code, ymd)
        row = _row_for_stock(rows, stock_code)
        close = _parse_close(row) if row else None
        return ymd, close

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(job, d): d for d in dates}
        for fut in as_completed(futs):
            try:
                ymd, close = fut.result()
                if close is not None:
                    results[ymd] = close
            except Exception:
                pass
    return results


def forward_returns_from_event_krx_api(
    stock_code: str,
    event_yyyymmdd: str,
) -> dict[str, Any]:
    """
    공시일(또는 이후 첫 거래일) 종가 대비 N거래일 후 전방 수익률(%).
    KRX OPEN API 일별매매정보만 사용.
    """
    digits = re.sub(r"\D", "", event_yyyymmdd)
    if len(digits) < 8:
        return {"error": "event_date 파싱 실패"}
    try:
        ev = datetime.strptime(digits[:8], "%Y%m%d").date()
    except ValueError:
        return {"error": "event_date 파싱 실패"}

    code6 = re.sub(r"\D", "", stock_code).zfill(6)
    if len(code6) != 6:
        return {"error": "stock_code는 6자리여야 합니다."}

    meta = krx_finder_stock(stock_code)
    mkt: Optional[str] = None
    if meta:
        mkt = str(meta.get("marketCode") or "STK")
    else:
        lq0 = latest_quote_krx(code6)
        if lq0.get("ok"):
            mkt = str(lq0.get("marketCode") or "STK")
        else:
            # 시장 미확정이어도 STK로 1차 조회 후 closes 비면 tools 쪽에서 Yahoo 보조 시세로 넘김
            mkt = "STK"
    closes: dict[str, float] = {}
    # 1차: 공시 전후 ~9개월(영업일) — 대부분 N거래일 오프셋 충족
    first_end = ev + timedelta(days=280)
    dates1 = _daterange_weekdays(ev - timedelta(days=21), first_end)
    closes.update(_prefetch_closes(code6, mkt, dates1))
    # 2차: 1Y(252거래일) 부족 시 말일까지 확장
    dates2 = _daterange_weekdays(first_end + timedelta(days=1), ev + timedelta(days=420))
    ymd_sorted = sorted(closes.keys())
    ev_s = ev.strftime("%Y%m%d")
    anchor_probe = next((y for y in ymd_sorted if y >= ev_s), None)
    ordered_probe = [y for y in ymd_sorted if y >= anchor_probe] if anchor_probe else []
    if dates2 and (not anchor_probe or len(ordered_probe) < 253):
        closes.update(_prefetch_closes(code6, mkt, dates2))
    if not dates1 and not dates2:
        return {"error": "조회 기간 없음"}
    if not closes:
        msg = "시세 응답이 비었습니다."
        if _krx_use_sample_path():
            msg += " 샘플 API는 일부 종목만 제공합니다. .env에 KRX_OPENAPI_KEY를 설정하세요."
        elif _krx_auth_key():
            msg += " KRX_OPENAPI_KEY 권한(유가증권·코스닥 일별매매정보 이용신청)을 확인하세요."
        return {"error": msg, "latest_quote": latest_quote_krx(code6)}

    return build_forward_return_dict(
        closes,
        ev,
        code6,
        source="KRX_OPENAPI",
        sample_mode=_krx_use_sample_path(),
        latest_quote_override=None,
        extra=None,
    )
