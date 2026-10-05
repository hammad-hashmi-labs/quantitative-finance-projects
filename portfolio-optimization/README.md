# Mean-Variance Portfolio Optimization

A constrained-optimization implementation of the Markowitz mean-variance
framework, built with `NumPy` and `SciPy.optimize`.

## What it does
- Simulates 1,000 days of returns for 5 assets (seeded, reproducible) from a
  multivariate normal distribution with realistic correlation structure
- Computes the **Minimum-Variance Portfolio** and **Maximum-Sharpe-Ratio
  Portfolio** using `scipy.optimize.minimize` (SLSQP) under a long-only,
  fully-invested constraint (weights ≥ 0, sum to 1)
- Benchmarks both against an equal-weight baseline

## Results (this run, seed=42)
| Portfolio | Annual Return | Volatility | Sharpe Ratio |
|---|---|---|---|
| Equal-Weight Baseline | 7.73% | 13.34% | 0.280 |
| Minimum-Variance | 1.69% | 4.77% | -0.484 |
| Maximum-Sharpe | 13.80% | 22.75% | **0.431** |

The optimizer improved the Sharpe ratio by **53.9%** over the naive
equal-weight baseline.

## Data note
Returns are **simulated** (not pulled from a live market feed), generated
from asset-specific mean/volatility assumptions and a specified correlation
matrix. This is a methodology demonstration of constrained portfolio
optimization, not a backtested trading strategy or a claim about real
historical performance.

## Run it
```bash
pip install numpy scipy
python portfolio_optimizer.py
```

## Techniques demonstrated
Constrained nonlinear optimization (SLSQP) · Covariance matrix construction ·
Sharpe ratio maximization · Efficient frontier concepts · NumPy vectorized
linear algebra
