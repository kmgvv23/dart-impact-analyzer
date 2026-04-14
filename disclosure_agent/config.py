import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env (다른 CWD에서 import 해도 동일 파일 로드)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
    return key


def get_dart_api_key() -> str:
    return os.getenv("DART_API_KEY", "").strip()
