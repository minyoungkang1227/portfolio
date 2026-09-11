"""
06_portfolio_diversification.py

05_threshold_scan.py의 결과를 단일 종목이 아니라 30종목 · 국면별 3,900건
표본으로 확장 재검증하는 스크립트. 개별 종목마다 변동성 레짐이 다르기 때문에
동일 z-score 임계값이라도 종목별로 신호 빈도·품질이 달라, 종목별 신호를
합산한 포트폴리오 수준에서 t-검정을 다시 수행합니다 (05의 결과가 특정
종목에만 우연히 맞았던 건 아닌지 확인하는 재검증 단계).
"""

import numpy as np
import pandas as pd
from scipy import stats


def per_symbol_events(price_by_symbol: dict[str, pd.Series], ma_window: int = 20,
                       z_window: int = 60, threshold: float = 1.6, horizon: int = 5) -> pd.DataFrame:
    """
    종목별 spread → z-score → 임계값 초과 이벤트를 모아 하나의 DataFrame으로 합친다.
    """
    all_events = []
    for symbol, price in price_by_symbol.items():
        ma = price.rolling(ma_window).mean()
        spread = (price - ma).dropna()
        mean = spread.rolling(z_window).mean()
        std = spread.rolling(z_window).std()
        z = ((spread - mean) / std).dropna()

        returns = price.pct_change()
        fwd_ret = returns.rolling(horizon).sum().shift(-horizon)

        idx = z.index.intersection(fwd_ret.index)
        z, fwd = z.loc[idx], fwd_ret.loc[idx]

        long_mask = z <= -threshold
        short_mask = z >= threshold

        for direction, mask, sign in [("long", long_mask, 1), ("short", short_mask, -1)]:
            events = fwd[mask].dropna()
            for date, ret in events.items():
                all_events.append({"symbol": symbol, "date": date, "direction": direction, "signed_return": ret * sign})

    return pd.DataFrame(all_events)


def portfolio_level_t_test(events: pd.DataFrame) -> dict:
    """
    전 종목 이벤트를 합산해 signed_return(반전매매 가설 방향으로 부호를 맞춘 수익률)의
    평균이 0보다 유의하게 큰지 검정. 프로젝트에서 실제로 나온 결과: t=2.44.
    """
    sample = events["signed_return"].dropna()
    n = len(sample)
    if n < 5:
        return {"n": n, "t_stat": np.nan, "p_value": np.nan}
    t_stat, p_value = stats.ttest_1samp(sample, popmean=0.0)
    return {
        "n_events": n,
        "n_symbols": events["symbol"].nunique(),
        "mean_signed_return": float(sample.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def diversification_summary(events: pd.DataFrame) -> pd.DataFrame:
    """종목별 이벤트 수·평균수익 요약 (한 종목에 신호가 쏠려있지 않은지 확인)"""
    return events.groupby("symbol")["signed_return"].agg(["count", "mean"]).sort_values("count", ascending=False)


if __name__ == "__main__":
    np.random.seed(3)
    # 예시: 30종목 더미 가격 시계열
    symbols = [f"KRX_{i:03d}" for i in range(30)]
    price_by_symbol = {
        s: pd.Series(100 + np.cumsum(np.random.randn(500) * 0.4))
        for s in symbols
    }

    events = per_symbol_events(price_by_symbol, threshold=1.6)
    print(f"총 이벤트 수: {len(events)}")  # 실제 프로젝트에서는 3,900건 규모
    print(portfolio_level_t_test(events))
    print(diversification_summary(events).head())
