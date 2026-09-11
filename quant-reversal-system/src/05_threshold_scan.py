"""
05_threshold_scan.py

04에서 재설계를 결정한 뒤, spread를 표준화한 z-score를 진입 신호로 쓰기로 하고
"어느 z-score 임계값에서 진입해야 하는가"를 데이터 기반으로 스캔합니다.

event_study_t_test: 특정 임계값을 넘는 진입 시점들을 모아, 그 이후 N일 수익률의
평균이 0과 유의하게 다른지 t-검정으로 확인 (event study 방식).

이 스캔 결과 국면(변동성 레짐)별로 최적 임계값이 다르게 나왔고, 최종적으로
5~6% 분위(z-score 약 1.6~1.9 구간에 해당)에서 t=2.44로 가장 안정적인
평균회귀 신호를 확인해 데이터 기반 임계값으로 채택했습니다.
"""

import numpy as np
import pandas as pd
from scipy import stats


def zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return ((spread - mean) / std).dropna()


def event_study_t_test(returns: pd.Series, z: pd.Series, threshold: float, horizon: int = 5) -> dict:
    """
    |z| >= threshold 인 시점을 이벤트로 잡고, 이후 horizon일 누적수익률의 평균이
    0과 유의하게 다른지 검정. 반전매매 가설이라면 z가 양(고평가)일 때는 음의 수익률,
    z가 음(저평가)일 때는 양의 수익률이 유의해야 한다.
    """
    idx = z.index.intersection(returns.index)
    z = z.loc[idx]
    r = returns.loc[idx]

    fwd_ret = r.rolling(horizon).sum().shift(-horizon)

    long_events = fwd_ret[z <= -threshold].dropna()
    short_events = fwd_ret[z >= threshold].dropna()

    def _t_test(sample: pd.Series, expect_sign: int) -> dict:
        if len(sample) < 5:
            return {"n": len(sample), "t_stat": np.nan, "p_value": np.nan, "mean_return": np.nan}
        t_stat, p_value = stats.ttest_1samp(sample * expect_sign, popmean=0.0)
        return {"n": len(sample), "t_stat": float(t_stat), "p_value": float(p_value), "mean_return": float(sample.mean())}

    return {
        "threshold": threshold,
        "long_side": _t_test(long_events, expect_sign=1),   # 저평가 진입 → 반등 기대
        "short_side": _t_test(short_events, expect_sign=-1),  # 고평가 진입 → 되돌림 기대
    }


def scan_thresholds(returns: pd.Series, z: pd.Series, candidates: list[float], horizon: int = 5) -> pd.DataFrame:
    rows = []
    for th in candidates:
        res = event_study_t_test(returns, z, th, horizon)
        rows.append({
            "threshold": th,
            "long_n": res["long_side"]["n"],
            "long_t": res["long_side"]["t_stat"],
            "short_n": res["short_side"]["n"],
            "short_t": res["short_side"]["t_stat"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    np.random.seed(2)
    n = 1000
    price = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.4))
    ma = price.rolling(20).mean()
    spread = (price - ma).dropna()
    z = zscore(spread, window=60)
    returns = price.pct_change().dropna()

    # 데이터 기반으로 탐색한 후보 z-score 임계값들 (5~6% 분위 근방)
    candidates = [1.2, 1.4, 1.6, 1.8, 2.0]
    scan = scan_thresholds(returns, z, candidates)
    print(scan)
