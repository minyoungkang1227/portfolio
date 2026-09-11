"""
compound_loss_simulation.py

제16회 DB보험금융공모전 "AI 리스크 보험 상품 모델링" 프로젝트의 핵심 모델링
코드입니다. AI 사고를 보험 손해의 관점에서 모델링하기 위해 복합포아송분포
(Compound Poisson Process)를 사용했습니다.

빈도-심도 분리 모델링:
  - 사고 발생 빈도  N(t) ~ Poisson(λ)         : 단위 기간당 사고 건수
  - 개별 손해액     X_i  ~ LogNormal(μ, σ)     : 사고 1건당 손해액 (Heavy tail 반영)
  - 총 손해액       S(t) = Σ_{i=1}^{N(t)} X_i   : 복합포아송과정

독립사건 가정(사고 발생 빈도와 개별 손해액이 서로 독립)하에 두 분포를
결합해 총손실 분포를 구성하고, 몬테카를로 시뮬레이션으로 분포를 근사합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompoundPoissonLogNormal:
    """
    N(t) ~ Poisson(lam), X_i ~ LogNormal(mu, sigma)
    S(t) = sum_{i=1}^{N(t)} X_i
    """
    lam: float          # 단위기간 사고 발생 빈도 (포아송 λ)
    mu: float            # 개별 손해액의 로그정규분포 위치모수
    sigma: float         # 개별 손해액의 로그정규분포 형태모수 (heavy tail 조절)

    def theoretical_moments(self) -> dict:
        """
        복합포아송과정의 이론적 평균·분산 (독립사건 가정하 결합):
            E[S]   = E[N] * E[X]
            Var[S] = E[N] * Var[X] + Var[N] * E[X]^2
        """
        e_x = np.exp(self.mu + self.sigma ** 2 / 2)
        var_x = (np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)

        e_s = self.lam * e_x
        var_s = self.lam * var_x + self.lam * e_x ** 2  # Var[N]=E[N]=lam (Poisson 특성)

        return {"E_S": float(e_s), "Var_S": float(var_s), "Std_S": float(np.sqrt(var_s))}

    def simulate(self, n_periods: int = 100_000, seed: int | None = None) -> np.ndarray:
        """
        몬테카를로: n_periods개의 독립적인 기간에 대해 S(t)를 반복 추출.
        각 기간마다 (1) 사고건수를 포아송에서 뽑고, (2) 건수만큼 로그정규 손해액을
        뽑아 합산한다. 이 반복 추출이 시뮬레이션 기반 총손실 분포 근사의 핵심.
        """
        rng = np.random.default_rng(seed)
        counts = rng.poisson(self.lam, size=n_periods)
        totals = np.empty(n_periods)

        for i, n in enumerate(counts):
            if n == 0:
                totals[i] = 0.0
                continue
            losses = rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)
            totals[i] = losses.sum()

        return totals


if __name__ == "__main__":
    # 예시 파라미터: 실제 프로젝트에서는 공모전 제공 사고 데이터로 lam/mu/sigma를 적률추정
    model = CompoundPoissonLogNormal(lam=3.2, mu=2.0, sigma=1.1)

    theo = model.theoretical_moments()
    print("이론적 모멘트:", theo)

    sims = model.simulate(n_periods=200_000, seed=42)
    print(f"시뮬레이션 평균: {sims.mean():.2f}, 표준편차: {sims.std():.2f}")
