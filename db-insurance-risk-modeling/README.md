# AI 리스크 보험 상품 모델링

제16회 DB보험금융공모전 참가 프로젝트 (모델링 담당). AI 시스템의 사고를 보험
손해의 관점에서 계량화하고, VaR 기반 필요자본금을 산출한 프로젝트입니다.

## 모델링 구조

1. **빈도-심도 분리 모델링** — 사고 발생 빈도는 포아송분포, 개별 손해액은
   로그정규분포로 각각 모델링 (독립사건 가정)
2. **총손실 분포 결합** — 복합포아송과정(Compound Poisson Process)으로
   결합해 `S(t) = Σ X_i, N(t) ~ Poisson(λ)`
3. **몬테카를로 시뮬레이션** — 반복 추출로 총손해액 분포를 근사, Heavy Tail
   구간에서 분산이 급증하는 걸 확인
4. **VaR 산출 & 민감도 분석** — 95%/99% 신뢰수준 VaR로 손실 규모를 정량화하고,
   `Required Capital = VaR₉₉%(S) - E[S]`로 필요 자본금 산출. λ·분포
   파라미터 변화에 따른 민감도를 스캔해 AI 사건화 리스크가 Tail 쪽에
   비선형적으로 반영된다는 걸 확인 → **Tail 중심 자본관리** 결론

## 파일

| 파일 | 내용 |
|---|---|
| `src/compound_loss_simulation.py` | 복합포아송·로그정규 결합 모델, 이론적 모멘트 계산, 몬테카를로 시뮬레이션 |
| `src/var_and_sensitivity.py` | VaR/필요자본금 산출, λ·σ 민감도 분석 |

## 실행

```bash
pip install -r requirements.txt
python src/var_and_sensitivity.py
```

`compound_loss_simulation.py`를 직접 실행하면 이론적 모멘트와 시뮬레이션
평균·표준편차가 서로 수렴하는지 확인할 수 있습니다(적률 검증).
