# 강민영 — 금융공학 · 리스크 모델링 포트폴리오

**한국어** | [English](README.en.md)

경희대학교 수학과. 확률 모델로 금융·보험 리스크를 계량하고, 그 결과를 **이론값 대조와 재검증으로 확인하는 방식**으로 프로젝트를 진행합니다. 확인되지 않은 가설은 과감히 재검토하거나 폐기합니다.

📄 **포트폴리오 PDF:** [docs/portfolio_minyoung_kang.pdf](docs/portfolio_minyoung_kang.pdf)

## 프로젝트

| 프로젝트 | 내용 | 핵심 기법 |
|---|---|---|
| **[quant-reversal-system](quant-reversal-system)** | 국내 주식 반전매매(평균회귀) 퀀트 시스템. 랜덤워크 sanity check로 가짜 평균회귀를 발견해 방법론을 폐기하고, 30종목·3,874건 사건으로 재검증 후 자동매매 구현 | OU 과정 MLE, t-검정, 임계치 스캔, 4단계 리스크 관리 |
| **[db-insurance-risk-modeling](db-insurance-risk-modeling)** | 제16회 DB보험금융공모전 "AI 리스크 보험 상품 모델링" | 복합포아송·음이항 빈도, 로그정규+파레토 심도, 몬테카를로, VaR 95%/99%, 민감도 분석 |
| **[pinescript-candle-signal](pinescript-candle-signal)** | 공개 캔들·거래량 지표를 분석해 설계한 결합 진입 필터 (원본 지표는 라이선스와 함께 별도 표기) | body/wick 비율, 거래량 델타, 추세 필터 |

## 공통 작업 방식

1. **이론부터 수치로 검증** — 모델을 쓰기 전에 시뮬레이션 결과가 이론값과 맞는지 먼저 확인합니다 (브라운 운동 분산, 이토 등거리, GBM 해석해 등).
2. **결과를 의심하고 재검증** — 그럴듯한 결과가 나와도 표본 편향, 방법론 자체의 함정, 과최적화를 먼저 점검합니다.
3. **한계를 숨기지 않기** — 각 프로젝트 README에 가정과 한계, 다음 검증 계획을 함께 적습니다.

## 기술

Python (NumPy, pandas, SciPy, Matplotlib) · Pine Script · SQL · 확률미분방정식 · 계리 모델링

## 연락처

- Email: doongss1@naver.com
