"""KRX Open API가 막히거나(403 등) 빈 응답일 때 Yahoo Finance 보조 시세."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

from disclosure_agent.krx_openapi import build_forward_return_dict


def _code6(stock_code: str) -> str:
    return re.sub(r"\D", "", stock_code).zfill(6)


def _pick_symbol(code6: str) -> tuple[Optional[str], Optional[str]]:
    """(yfinance 심볼, STK|KSQ) — .KS 우선 후 .KQ."""
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    for suf, mkt in ((".KS", "STK"), (".KQ", "KSQ")):
        sym = f"{code6}{suf}"
        t = yf.Ticker(sym)
        h = t.history(period="5d", auto_adjust=False)
        if h is not None and len(h) > 0:
            return sym, mkt
    return None, None


def latest_quote_yfinance(stock_code: str) -> dict[str, Any]:
    code6 = _code6(stock_code)
    if len(code6) != 6:
        return {"ok": False, "error": "stock_code는 6자리여야 합니다."}
    sym, mkt = _pick_symbol(code6)
    if not sym:
        return {"ok": False, "error": "Yahoo Finance에서 해당 종목을 찾지 못했습니다."}
    import yfinance as yf

    t = yf.Ticker(sym)
    hist = t.history(period="10d", auto_adjust=False)
    if hist is None or hist.empty:
        return {"ok": False, "error": "Yahoo Finance 시세가 비었습니다."}
    idx = hist.index[-1]
    ymd = idx.strftime("%Y%m%d")
    close = float(hist["Close"].iloc[-1])
    return {
        "ok": True,
        "basDd": ymd,
        "marketCode": mkt or "STK",
        "close": close,
        "isu_nm": sym,
        "source": "YAHOO_FINANCE",
        "sample_mode": False,
    }


def forward_returns_from_event_yfinance(
    stock_code: str,
    event_yyyymmdd: str,
) -> dict[str, Any]:
    digits = re.sub(r"\D", "", event_yyyymmdd)
    if len(digits) < 8:
        return {"error": "event_date 파싱 실패"}
    try:
        ev = datetime.strptime(digits[:8], "%Y%m%d").date()
    except ValueError:
        return {"error": "event_date 파싱 실패"}
    code6 = _code6(stock_code)
    if len(code6) != 6:
        return {"error": "stock_code는 6자리여야 합니다."}
    sym, _mkt = _pick_symbol(code6)
    if not sym:
        return {"error": "Yahoo Finance에서 해당 종목을 찾지 못했습니다."}
    try:
        import yfinance as yf
    except ImportError as e:
        return {"error": f"yfinance 미설치: {e}"}

    t = yf.Ticker(sym)
    start = ev - timedelta(days=45)
    end = ev + timedelta(days=420)
    hist = t.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if hist is None or hist.empty:
        return {"error": "Yahoo Finance 시세 이력이 비었습니다."}
    closes: dict[str, float] = {}
    for idx, row in hist.iterrows():
        ymd = idx.strftime("%Y%m%d")
        closes[ymd] = float(row["Close"])
    last_i = hist.index[-1]
    lq: dict[str, Any] = {
        "ok": True,
        "basDd": last_i.strftime("%Y%m%d"),
        "marketCode": _mkt or "STK",
        "close": float(hist["Close"].iloc[-1]),
        "isu_nm": sym,
        "source": "YAHOO_FINANCE",
        "sample_mode": False,
    }
    return build_forward_return_dict(
        closes,
        ev,
        code6,
        source="YAHOO_FINANCE",
        sample_mode=False,
        latest_quote_override=lq,
        extra={"price_source_note": "거래소 Open API를 쓰지 못해 Yahoo Finance로 계산했습니다. 참고용입니다."},
    )
