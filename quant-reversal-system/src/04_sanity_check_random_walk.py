"""
04_sanity_check_random_walk.py

프로젝트에서 실제로 방법론을 한 번 뒤집었던 지점입니다.
처음 세운 가설은 "spread = price - MA는 평균회귀 신호다" 였는데,
이 sanity check을 돌려보니 표본이 작을 때는 순수 랜덤워크도 겉보기에는
비슷한 패턴을 보인다는 게 확인됐습니다 (가짜 신호 / look-ahead bias 위험).

그래서 이 스크립트는 "spread가 진짜 평균회귀인지, 아니면 그냥 랜덤워크의
우연한 모양인지"를 몬테카를로 시뮬레이션으로 재검증합니다.
"""

import numpy as np
import pandas as pd


def simulate_random_walk(n: int, sigma: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, sigma, size=n)
    return np.cumsum(steps)


def lag1_autocorr(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def sanity_check(spread: pd.Series, n_sims: int = 1000, sigma: float | None = None, seed: int = 42) -> dict:
    """
    실제 spread의 1차 자기상관이, 같은 길이·변동성의 순수 랜덤워크 시뮬레이션 분포에서
    "통계적으로 유의하게" 벗어나는지 확인한다.
    (첫 가설: spread = price - MA 는 평균회귀 신호다)
    """
    spread = spread.values if isinstance(spread, pd.Series) else np.asarray(spread)
    n = len(spread)
    sigma = sigma if sigma is not None else float(np.diff(spread).std())

    observed_rho = lag1_autocorr(spread)

    rng = np.random.default_rng(seed)
    sim_rhos = np.empty(n_sims)
    for i in range(n_sims):
        rw = simulate_random_walk(n, sigma=sigma, seed=rng.integers(0, 1_000_000))
        sim_rhos[i] = lag1_autocorr(rw)

    p_value = float(np.mean(sim_rhos >= observed_rho))  # 관측값이 랜덤워크 분포 상위 몇 %인지
    is_significant = p_value < 0.05

    if not is_significant:
        # 가짜 신호 → 방법론 폐기 (재설계로 넘어가는 근거)
        verdict = "REJECT: random-walk-like, spread 단순 회귀 가설 폐기"
    else:
        verdict = "ACCEPT: random walk 대비 유의한 평균회귀 특성 확인"

    return {
        "observed_rho": observed_rho,
        "sim_rho_mean": float(sim_rhos.mean()),
        "sim_rho_std": float(sim_rhos.std()),
        "p_value": p_value,
        "is_significant": is_significant,
        "verdict": verdict,
    }


if __name__ == "__main__":
    # 실제 프로젝트에서 나온 재설계 근거: 05_threshold_scan.py에서 국면별 데이터 기반
    # 임계치(5~6%)를 채택하고, 06에서 30종목·3,900건으로 표본을 확장해 재검증함
    np.random.seed(0)
    fake_random_walk = pd.Series(simulate_random_walk(300, sigma=1.0, seed=0))
    print("random walk 자체 검증 (기대: REJECT):")
    print(sanity_check(fake_random_walk))
