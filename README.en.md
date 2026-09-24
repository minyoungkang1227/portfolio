# Minyoung Kang — Financial Engineering & Risk Modeling Portfolio

[한국어](README.md) | **English**

Mathematics major at Kyung Hee University. I quantify financial and insurance risk with probabilistic models and **check the results against theory and through re-validation**. Hypotheses that do not hold up are revisited or discarded.

📄 **Portfolio PDF (Korean):** [docs/portfolio_minyoung_kang.pdf](docs/portfolio_minyoung_kang.pdf)

## Projects

| Project | Description | Key methods |
|---|---|---|
| **[quant-reversal-system](quant-reversal-system)** | Short-term mean-reversion strategy for Korean equities. A random-walk sanity check revealed a spurious mean-reversion signal, so the method was discarded; the strategy was re-validated on 3,874 events across 30 stocks and automated | OU-process MLE, t-tests, threshold scans, four-layer risk controls |
| **[db-insurance-risk-modeling](db-insurance-risk-modeling)** | "AI risk insurance product modeling" for the 16th DB Insurance Finance Competition | Compound Poisson / negative binomial frequency, lognormal + Pareto severity, Monte Carlo, VaR 95%/99%, sensitivity analysis |
| **[pinescript-candle-signal](pinescript-candle-signal)** | Combined entry filter designed after analyzing public candle and volume indicators (original indicators credited with their licenses) | Body/wick ratios, volume delta, trend filters |

## How I work

1. **Verify theory numerically first** — before using a model, confirm that simulations match theoretical values (Brownian-motion variance, Itô isometry, closed-form GBM).
2. **Doubt the result and re-validate** — even when results look good, check for sample bias, pitfalls in the method itself, and overfitting.
3. **State the limitations** — each project README lists its assumptions, limitations and next validation steps.

## Skills

Python (NumPy, pandas, SciPy, Matplotlib) · Pine Script · SQL · Stochastic differential equations · Actuarial modeling

## Contact

- Email: doongss1@naver.com
