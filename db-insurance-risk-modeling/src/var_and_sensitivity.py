"""
var_and_sensitivity.py

compound_loss_simulation.py에서 만든 총손실 분포 시뮬레이션 결과로부터
VaR(Value at Risk)을 산출하고, 필요 자본금(Required Capital)을 계산합니다.
또한 λ(사고 발생 빈도)와 분포 파라미터가 변할 때 VaR이 얼마나 민감하게
반응하는지 민감도 분석을 수행합니다 (AI 사건화 리스크 → Tail 중심 자본관리 결론).

VaR_alpha(S)      = 손실분포 S의 alpha분위수 (예: 95%, 99% 신뢰수준)
Required Capital  = VaR_alpha(S) - E[S]   (기대손실을 초과하는 부분만큼 자본을 쌓아야 한다는 관점)
"""

from __future__ import annotations

import numpy as np

from compound_loss_simulation import CompoundPoissonLogNormal


def value_at_risk(sims: np.ndarray, alpha: float = 0.95) -> float:
    return float(np.quantile(sims, alpha))


def required_capital(sims: np.ndarray, alpha: float = 0.95) -> float:
    var = value_at_risk(sims, alpha)
    expected_loss = sims.mean()
    return var - expected_loss


def var_report(sims: np.ndarray) -> dict:
    return {
        "E[S]": float(sims.mean()),
        "VaR_95": value_at_risk(sims, 0.95),
        "VaR_99": value_at_risk(sims, 0.99),
        "RequiredCapital_95": required_capital(sims, 0.95),
        "RequiredCapital_99": required_capital(sims, 0.99),
    }


def sensitivity_to_lambda(base_mu: float, base_sigma: float, lam_grid: list[float],
                           n_periods: int = 100_000, seed: int = 0) -> list[dict]:
    """
    사고 발생 빈도(λ)가 변할 때 VaR_99가 얼마나 민감하게 움직이는지 스캔.
    AI 시스템의 사고 빈도가 예측보다 높아지는 시나리오(모델 드리프트, 신규 리스크
    유형 등)에 자본금이 얼마나 취약한지 확인하기 위한 분석.
    """
    results = []
    for lam in lam_grid:
        model = CompoundPoissonLogNormal(lam=lam, mu=base_mu, sigma=base_sigma)
        sims = model.simulate(n_periods=n_periods, seed=seed)
        report = var_report(sims)
        report["lambda"] = lam
        results.append(report)
    return results


def sensitivity_to_tail(base_lam: float, base_mu: float, sigma_grid: list[float],
                         n_periods: int = 100_000, seed: int = 0) -> list[dict]:
    """
    개별 손해액 분포의 형태모수(sigma, heavy tail 정도)가 변할 때 VaR_99 민감도.
    sigma가 커질수록 극단 손실(Tail risk)의 영향이 커지는데, 이게 필요자본에
    얼마나 비선형적으로 반영되는지가 "Tail 중심 자본관리"로 이어지는 근거.
    """
    results = []
    for sigma in sigma_grid:
        model = CompoundPoissonLogNormal(lam=base_lam, mu=base_mu, sigma=sigma)
        sims = model.simulate(n_periods=n_periods, seed=seed)
        report = var_report(sims)
        report["sigma"] = sigma
        results.append(report)
    return results


if __name__ == "__main__":
    base_model = CompoundPoissonLogNormal(lam=3.2, mu=2.0, sigma=1.1)
    sims = base_model.simulate(n_periods=200_000, seed=42)

    print("기본 VaR 리포트:")
    print(var_report(sims))

    print("\nλ 민감도 분석 (사고 발생 빈도 증가 시나리오):")
    for row in sensitivity_to_lambda(base_mu=2.0, base_sigma=1.1, lam_grid=[2.0, 3.2, 4.5, 6.0]):
        print(f"  λ={row['lambda']:.1f} -> VaR99={row['VaR_99']:.1f}, RequiredCapital99={row['RequiredCapital_99']:.1f}")

    print("\nTail(σ) 민감도 분석 (극단 손실 반영 정도):")
    for row in sensitivity_to_tail(base_lam=3.2, base_mu=2.0, sigma_grid=[0.7, 1.1, 1.5, 1.9]):
        print(f"  σ={row['sigma']:.1f} -> VaR99={row['VaR_99']:.1f}, RequiredCapital99={row['RequiredCapital_99']:.1f}")
