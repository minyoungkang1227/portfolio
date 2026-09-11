"""
01_sde_theory_verification.py

반전매매(평균회귀) 전략을 세우기 전에, "가격과 이동평균의 괴리(spread)가
정말로 평균회귀 과정을 따르는가?"를 확률과정 이론 수준에서 먼저 검증하는
스크립트입니다.

핵심 아이디어:
  - spread_t = price_t - MA_t 를 오른스타인-울렌벡(OU) 과정의 이산화 버전으로 가정
      d(spread) = theta * (mu - spread) * dt + sigma * dW
  - OU 과정이라면 spread의 1차 자기상관계수(rho)가 0 < rho < 1 이어야 하고,
    분산비(variance ratio)가 시간에 따라 1보다 작게 수렴해야 함 (순수 랜덤워크는 분산비 ≈ 1)
  - 이 스크립트는 그 두 가지를 계산해서 "평균회귀 가설"을 최초로 스크리닝합니다.
"""

import numpy as np
import pandas as pd


def compute_spread(price: pd.Series, ma_window: int = 20) -> pd.Series:
    ma = price.rolling(ma_window).mean()
    return (price - ma).dropna()


def lag1_autocorrelation(spread: pd.Series) -> float:
    x = spread.values[:-1]
    y = spread.values[1:]
    return float(np.corrcoef(x, y)[0, 1])


def variance_ratio(spread: pd.Series, k: int = 5) -> float:
    """
    Lo-MacKinlay 스타일의 분산비. k기간 수익률 분산을 1기간 수익률 분산의 k배와 비교.
    랜덤워크면 VR ≈ 1, 평균회귀면 VR < 1.
    """
    diffs_1 = spread.diff().dropna()
    diffs_k = spread.diff(k).dropna()
    var_1 = diffs_1.var()
    var_k = diffs_k.var()
    if var_1 == 0:
        return np.nan
    return float(var_k / (k * var_1))


def screen_mean_reversion(price: pd.Series, ma_window: int = 20, vr_k: int = 5) -> dict:
    spread = compute_spread(price, ma_window)
    rho = lag1_autocorrelation(spread)
    vr = variance_ratio(spread, vr_k)

    # 스크리닝 기준: 자기상관이 0~1 사이(평균회귀 방향)이고, 분산비가 1보다 뚜렷이 작을 것
    is_candidate = (0.0 < rho < 1.0) and (vr < 0.85)

    return {
        "lag1_autocorr": rho,
        "variance_ratio_k": vr,
        "is_mean_reversion_candidate": is_candidate,
    }


if __name__ == "__main__":
    # 예시: 실제 사용 시 OHLCV에서 종가 시계열을 price로 넣는다
    np.random.seed(0)
    dummy_price = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5))
    result = screen_mean_reversion(dummy_price)
    print(result)
