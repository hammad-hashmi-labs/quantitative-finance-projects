# Mean-Variance Portfolio Optimization (with Out-of-Sample Evaluation)

A constrained-optimization implementation of the Markowitz mean-variance
framework, built with `NumPy` and `SciPy.optimize` and tested the way a
portfolio would actually be used: weights are **fit on a training window and
judged on later, unseen returns**.

## What it does
- Simulates 1,000 days of returns for 5 assets (seeded, reproducible) from a
  multivariate normal distribution with a specified correlation structure
- Splits the data **chronologically**: the first 70% (700 days) is used to fit,
  the last 30% (300 days) to test. The split is never shuffled.
- Fits three strategies on the training window only: an **equal-weight
  baseline**, the **minimum-variance** portfolio, and the **maximum-Sharpe**
  portfolio (`scipy.optimize.minimize`, SLSQP, long-only, fully invested)
- Reports in-sample and out-of-sample Sharpe ratios (risk-free rate 4%)
- Repeats the whole fit-then-test procedure over **500 independently simulated
  paths** to separate real effects from single-path luck

## Results

### Single path (seed 42)
| Strategy | In-sample Sharpe | Out-of-sample Sharpe |
|---|---|---|
| Equal-Weight | -0.129 | 1.240 |
| Min-Variance | -1.374 | 1.747 |
| Max-Sharpe | 0.326 | 0.443 |

On this path the max-Sharpe optimizer put 100% of the portfolio in a single
asset (`Equity_B`), a corner solution driven by noisy estimated returns. A
300-day test window is short and noisy, so one path says little on its own;
the 500-path table below is the more reliable summary.

### 500 simulated paths (seeds 1000-1499)
| Strategy | Median in-sample Sharpe | Median out-of-sample Sharpe |
|---|---|---|
| Equal-Weight | 0.372 | 0.365 |
| Min-Variance | 0.083 | 0.066 |
| Max-Sharpe | 0.908 | 0.279 |

- Max-Sharpe beat the equal-weight baseline out-of-sample in **45.6%** of paths
  (median Sharpe difference: -0.072)
- Median in-sample-to-out-of-sample Sharpe decay for Max-Sharpe: **0.648**
- The solver converged on every fit (0 failures in 500 paths)

## Interpretation
- The max-Sharpe portfolio looks far better on the data it was fit to (median
  Sharpe 0.908) than on unseen data (0.279). Most of the in-sample advantage
  comes from fitting noise in the estimated mean returns.
- Out-of-sample, it showed no reliable edge over simple equal weighting.
- This is the well-known *estimation error* problem in mean-variance
  optimization (Michaud, 1989), and it is consistent with evidence that
  equal-weight diversification is hard to beat out-of-sample (DeMiguel,
  Garlappi & Uppal, 2009).
- A natural next step would be to test mitigations such as shrinkage
  covariance estimates, weight caps, or rolling re-estimation.

## Data note
Returns are **simulated** (seeded for reproducibility) from asset-specific
mean/volatility assumptions and a specified correlation matrix. This project
demonstrates the methodology and the effect of estimation error. It is not a
backtest of a trading strategy and makes no claim about real historical
market performance.

## Run it
```bash
pip install numpy scipy
python portfolio_optimizer.py
```

## Techniques demonstrated
Constrained nonlinear optimization (SLSQP) · Covariance estimation ·
Sharpe ratio maximization · Chronological train/test validation ·
Monte Carlo robustness testing · NumPy vectorized linear algebra
