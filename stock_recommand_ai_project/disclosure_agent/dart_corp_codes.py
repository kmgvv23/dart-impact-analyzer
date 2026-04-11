"""
DART 고유번호(corp_code) — Open DART `corpCode.xml`(ZIP) 전체 목록 캐시 후 회사명 매칭.

`company.json`은 corp_code 필수라 회사명 검색에 사용할 수 없음.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import requests

from disclosure_agent.config import get_dart_api_key

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / ".cache"
_CACHE_JSON = _CACHE_DIR / "dart_corp_list.json"
_CACHE_META = _CACHE_DIR / "dart_corp_list.meta.json"
_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
_MAX_CACHE_SEC = int(os.getenv("DART_CORP_LIST_CACHE_SEC", str(24 * 3600)))


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _xml_to_rows(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    rows: list[dict[str, str]] = []
    for el in root.iter("list"):
        corp_code = (el.findtext("corp_code") or "").strip()
        corp_name = (el.findtext("corp_name") or "").strip()
        stock_code = (el.findtext("stock_code") or "").strip()
        if not corp_code or not corp_name:
            continue
        rows.append(
            {
                "corp_code": corp_code,
                "corp_name": corp_name,
                "stock_code": stock_code,
            }
        )
    return rows


def _fetch_corp_zip_bytes(api_key: str) -> bytes:
    r = requests.get(_CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=120)
    r.raise_for_status()
    return r.content


def _extract_xml_from_response(content: bytes) -> bytes:
    c = content.lstrip(b"\xef\xbb\xbf")
    if len(c) >= 2 and c[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".xml"):
                    return zf.read(name)
        raise ValueError("ZIP 안에 XML이 없습니다.")
    if c.strip().startswith(b"<?xml") or c.strip().startswith(b"<"):
        root = ET.fromstring(c)
        st = root.findtext(".//status") or root.findtext("status")
        if st and st != "000":
            msg = root.findtext(".//message") or root.findtext("message") or st
            raise RuntimeError(f"DART corpCode.xml 오류: {msg}")
        return c
    raise ValueError("알 수 없는 corpCode.xml 응답 형식입니다.")


def _load_rows_from_cache() -> Optional[list[dict[str, str]]]:
    if not _CACHE_JSON.is_file():
        return None
    try:
        meta = json.loads(_CACHE_META.read_text(encoding="utf-8"))
        fetched_at = float(meta.get("fetched_at", 0))
        if time.time() - fetched_at > _MAX_CACHE_SEC:
            return None
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    try:
        data = json.loads(_CACHE_JSON.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data  # type: ignore[return-value]
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _save_cache(rows: list[dict[str, str]]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_JSON.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    _CACHE_META.write_text(json.dumps({"fetched_at": time.time()}, ensure_ascii=False), encoding="utf-8")


def get_corp_rows() -> list[dict[str, str]]:
    """전체 공시대상 법인 목록(캐시 우선)."""
    cached = _load_rows_from_cache()
    if cached is not None:
        return cached
    key = get_dart_api_key()
    if not key:
        return []
    try:
        raw = _fetch_corp_zip_bytes(key)
        xml_bytes = _extract_xml_from_response(raw)
        rows = _xml_to_rows(xml_bytes)
    except Exception:
        return []
    if rows:
        _save_cache(rows)
    return rows


def resolve_corp_by_name(query: str) -> Optional[dict[str, Any]]:
    """
    회사명(또는 정식명칭 일부)으로 상장사 우선 1건 선택.
    반환: {corp_code, corp_name, stock_code}
    """
    q = (query or "").strip()
    if len(q) < 2:
        return None
    qn = _norm(q)
    rows = get_corp_rows()
    if not rows:
        return None

    listed = [r for r in rows if (r.get("stock_code") or "").strip()]
    pool = listed if listed else rows

    exact = [r for r in pool if _norm(r["corp_name"]) == qn]
    if exact:
        exact.sort(key=lambda r: len(r["corp_name"]))
        pick = exact[0]
        return {**pick, "stock_code": (pick.get("stock_code") or "").zfill(6)}

    contains = [r for r in pool if qn in _norm(r["corp_name"])]
    if contains:
        contains.sort(key=lambda r: (len(_norm(r["corp_name"])) - len(qn), r["corp_name"]))
        pick = contains[0]
        return {**pick, "stock_code": (pick.get("stock_code") or "").zfill(6)}

    return None
