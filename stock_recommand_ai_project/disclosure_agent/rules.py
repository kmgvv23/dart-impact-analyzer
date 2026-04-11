"""룰 기반 공시 유형·회색지대 분류 (seed / PRD)."""

import re
from typing import Optional


def collapse_title_ws(s: str) -> str:
    """DART 제목·본문 앞부분의 공백·개행 제거(삼성전자식 '연결 및 별도 …' 매칭용)."""
    return re.sub(r"\s+", "", (s or ""))


GREY_KEYWORDS = (
    "정정",
    "기재정정",
    "정정신고",
    "정정보고",
    "첨부정정",
    "자진정정",
)

TARGET_TYPES = (
    "유상증자",
    "CB_BW",
    "잠정실적",
    "합병분할",
    "최대주주변경",
    "대규모계약",
    "관리종목_상장폐지",
)

KEYWORD_MAP: list[tuple[str, str]] = [
    ("유상증자", "유상증자"),
    ("유상 증자", "유상증자"),
    ("전환사채", "CB_BW"),
    ("신주인수권부사채", "CB_BW"),
    ("BW", "CB_BW"),
    ("CB", "CB_BW"),
    ("잠정실적", "잠정실적"),
    ("실적공시", "잠정실적"),
    ("연결실적", "잠정실적"),
    ("분기실적", "잠정실적"),
    ("반기실적", "잠정실적"),
    # 삼성전자 등 자율공시: 제목에 '잠정실적' 문구 없이 '영업실적…전망'만 쓰는 경우
    ("영업실적등에대한전망", "잠정실적"),
    ("영업실적에대한전망", "잠정실적"),
    ("실적등에대한전망", "잠정실적"),
    ("영업실적등에관한전망", "잠정실적"),
    ("연결및별도재무제표기준영업실적", "잠정실적"),
    ("연결ㆍ별도재무제표기준영업실적", "잠정실적"),
    ("합병", "합병분할"),
    ("분할", "합병분할"),
    ("분할합병", "합병분할"),
    ("최대주주변경", "최대주주변경"),
    ("최대주주", "최대주주변경"),
    ("대규모계약", "대규모계약"),
    ("단일판매", "대규모계약"),
    ("공급계약", "대규모계약"),
    ("관리종목", "관리종목_상장폐지"),
    ("상장폐지", "관리종목_상장폐지"),
    ("상장적격성", "관리종목_상장폐지"),
]


def is_grey_zone(title: str, body: str) -> bool:
    text = f"{collapse_title_ws(title)}\n{collapse_title_ws(body[:2000])}"
    return any(k in text for k in GREY_KEYWORDS)


def classify_disclosure_type(title: str, body: str) -> Optional[str]:
    """분석 대상 유형 코드 또는 None(비대상)."""
    text = f"{collapse_title_ws(title)}\n{collapse_title_ws(body[:8000])}"
    hits: list[str] = []
    for kw, code in KEYWORD_MAP:
        if kw in text:
            hits.append(code)
    if not hits:
        return None
    priority = [
        "관리종목_상장폐지",
        "유상증자",
        "CB_BW",
        "합병분할",
        "최대주주변경",
        "대규모계약",
        "잠정실적",
    ]
    for p in priority:
        if p in hits:
            return p
    return hits[0]


def extract_rcept_no(user_text: str) -> Optional[str]:
    m = re.search(r"rcpNo=(\d{14})", user_text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"rcept_no\D*(\d{14})", user_text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{14})\b", user_text)
    if m:
        return m.group(1)
    return None
