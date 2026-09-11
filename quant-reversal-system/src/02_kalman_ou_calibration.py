"""
02_kalman_ou_calibration.py

01_sde_theory_verification.py에서 평균회귀 후보로 판정된 spread에 대해,
오른스타인-울렌벡(OU) 과정의 파라미터 theta(회귀 속도), mu(장기평균),
sigma(변동성)를 추정합니다.

OU 과정의 이산화 형태는 AR(1) 회귀와 동치이므로:
    spread_t = a + b * spread_{t-1} + eps_t
    theta = -ln(b) / dt
    mu    = a / (1 - b)
    sigma = std(eps) * sqrt(2*theta / (1 - b^2))

여기서는 배치 OLS로 AR(1) 계수를 우선 추정하고, 시계열이 non-stationary한
구간을 다루기 위해 간단한 칼만 필터(스칼라 상태공간)로 파라미터를 순차
업데이트하는 버전도 함께 제공합니다.
"""

import numpy as np
import pandas as pd


def ar1_ols_calibration(spread: pd.Series, dt: float = 1.0) -> dict:
    y = spread.values[1:]
    x = spread.values[:-1]
    X = np.vstack([np.ones_like(x), x]).T

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    resid = y - X @ coef

    b = np.clip(b, 1e-6, 0.999999)  # theta가 정의되려면 0 < b < 1
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = resid.std(ddof=2) * np.sqrt(2 * theta / (1 - b ** 2))

    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma), "ar1_b": float(b)}


class ScalarKalmanOU:
    """
    상태 x_t = spread_t, 관측값 = spread_t (노이즈 있는 관측 가정).
    전이: x_t = mu + b*(x_{t-1} - mu) + w_t,  w_t ~ N(0, Q)
    관측: z_t = x_t + v_t,                    v_t ~ N(0, R)

    OU 파라미터(theta, mu, sigma)를 순차적으로 재추정하면서 상태를 필터링합니다.
    장기간 데이터에서 파라미터가 서서히 변하는(regime shift) 상황에 대응하기 위한 버전.
    """

    def __init__(self, dt: float = 1.0, window: int = 60):
        self.dt = dt
        self.window = window

    def run(self, spread: pd.Series) -> pd.DataFrame:
        values = spread.values
        n = len(values)
        filtered = np.zeros(n)
        theta_series = np.zeros(n)

        filtered[0] = values[0]
        for t in range(1, n):
            lo = max(0, t - self.window)
            window_slice = pd.Series(values[lo:t + 1])
            if len(window_slice) < 10:
                filtered[t] = values[t]
                continue
            params = ar1_ols_calibration(window_slice, dt=self.dt)
            theta_series[t] = params["theta"]

            # 칼만 예측/보정 (관측 노이즈 R, 프로세스 노이즈 Q는 파라미터 기반 근사)
            b = np.exp(-params["theta"] * self.dt)
            pred = params["mu"] + b * (filtered[t - 1] - params["mu"])
            Q = params["sigma"] ** 2 * (1 - b ** 2) / (2 * params["theta"] + 1e-9)
            R = Q * 0.5  # 관측 노이즈를 프로세스 노이즈 대비 절반 수준으로 가정
            K = Q / (Q + R)
            filtered[t] = pred + K * (values[t] - pred)

        return pd.DataFrame({"spread": values, "filtered": filtered, "theta": theta_series})


if __name__ == "__main__":
    np.random.seed(1)
    n = 500
    theta_true, mu_true, sigma_true = 0.08, 0.0, 1.2
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] + theta_true * (mu_true - spread[t - 1]) + sigma_true * np.random.randn()
    spread = pd.Series(spread)

    print("OLS calibration:", ar1_ols_calibration(spread))

    kf = ScalarKalmanOU(window=60)
    out = kf.run(spread)
    print(out.tail())
