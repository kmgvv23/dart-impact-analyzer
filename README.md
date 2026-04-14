# dart-impact-analyzer

> DART 공시를 입력하면 영향도 판단, 유사 사례 분석, 주가 패턴까지 분석해주는 한국 공시 투자 판단 AI 에이전트

개인 투자자가 법률·회계 용어로 가득한 공시를 빠르게 이해하고 투자 의사결정을 할 수 있도록 돕는 멀티에이전트 AI 서비스입니다.

## 주요 기능

### 공시 투자 판단
공시 URL 또는 회사명을 입력하면 자동으로 분석합니다.

| 단계 | 내용 |
|---|---|
| **공시 수집** | DART API로 공시 원문 자동 조회 |
| **유형 분류** | 유상증자·CB/BW·잠정실적·합병·최대주주변경 등 7가지 자동 분류 |
| **영향도 판단** | HIGH / MEDIUM / LOW 3단계 판정 + 근거 |
| **유사 사례** | 동업종·동일 유형 참고 사례 분석 |
| **주가 패턴** | 공시 전후 1주·1개월·3개월·6개월·1년 수익률 계산 |
| **투자 리포트** | Base/Best/Worst 시나리오 + 손절/익절/진입/관망 액션 |

### 재무 데이터 챗봇
자연어로 재무 데이터를 질의하면 표와 차트로 답변합니다.
> 예: "삼성전자 10개년 매출액 보여줘"

## 설치 및 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 실행
streamlit run streamlit_app.py
```

## 필요한 API 키

`.env` 파일을 생성하고 아래 키를 입력하세요.

```env
OPENAI_API_KEY=your_key
DART_API_KEY=your_key
KRX_OPENAPI_KEY=your_key  # 선택
```

| 환경변수 | 발급처 |
|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com |
| `DART_API_KEY` | https://opendart.fss.or.kr |
| `KRX_OPENAPI_KEY` | https://openapi.krx.co.kr |

## 기술 스택

| 구분 | 기술 |
|---|---|
| 에이전트 프레임워크 | LangGraph (Supervisor 패턴) |
| LLM | GPT-4o-mini |
| UI | Streamlit |
| 공시 데이터 | DART Open API |
| 주가 데이터 | KRX Open API + Yahoo Finance |
| 웹 검색 | DuckDuckGo (컨센서스 조회) |

## 분석 대상 공시 유형

- 유상증자 (희석 리스크)
- CB / BW (전환사채 / 신주인수권부사채)
- 잠정실적 (어닝 서프라이즈 / 쇼크)
- 합병 / 분할
- 최대주주변경
- 대규모계약
- 관리종목 / 상장폐지

## 주의사항

본 서비스는 투자 참고용 정보 제공을 목적으로 하며, 투자 권유나 매매 추천이 아닙니다. 모든 투자 결정은 본인의 책임입니다.

## License

MIT
