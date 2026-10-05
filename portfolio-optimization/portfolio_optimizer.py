"""
Mean-Variance Portfolio Optimization (Markowitz Model)
--------------------------------------------------------
Computes the Minimum-Variance Portfolio and the Maximum-Sharpe-Ratio
Portfolio across a set of assets using constrained optimization
(scipy.optimize.minimize, SLSQP).

NOTE ON DATA: This project uses SIMULATED daily return data
(seeded for reproducibility), generated from a multivariate normal
distribution with asset-specific mean/volatility assumptions loosely
representative of equity, bond, and commodity return profiles. It is
a methodology demonstration, not a claim about real historical market
performance or live trading returns.
"""

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------
# 1. Simulate daily returns for 5 synthetic assets
# ---------------------------------------------------------------
np.random.seed(42)

ASSET_NAMES = ["Equity_A", "Equity_B", "Bond_A", "Commodity_A", "Equity_C"]
N_ASSETS = len(ASSET_NAMES)
N_DAYS = 1000  # ~4 trading years

# Annualized assumptions (mean return, volatility) per asset
annual_mu    = np.array([0.10, 0.12, 0.04, 0.07, 0.14])
annual_sigma = np.array([0.18, 0.22, 0.05, 0.20, 0.28])

# Correlation matrix (symmetric, positive semi-definite by construction)
corr = np.array([
    [1.00, 0.65, 0.10, 0.20, 0.55],
    [0.65, 1.00, 0.05, 0.15, 0.60],
    [0.10, 0.05, 1.00, 0.00, 0.08],
    [0.20, 0.15, 0.00, 1.00, 0.18],
    [0.55, 0.60, 0.08, 0.18, 1.00],
])

daily_mu = annual_mu / 252
daily_sigma = annual_sigma / np.sqrt(252)
cov_daily = np.outer(daily_sigma, daily_sigma) * corr

returns = np.random.multivariate_normal(daily_mu, cov_daily, size=N_DAYS)

mean_returns = returns.mean(axis=0) * 252          # annualized
cov_matrix   = np.cov(returns, rowvar=False) * 252  # annualized

RISK_FREE_RATE = 0.04


def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return port_return, port_vol


def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(port_return - risk_free_rate) / port_vol


def portfolio_variance(weights, mean_returns, cov_matrix):
    _, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return port_vol ** 2


def optimize_portfolio(objective_fn, mean_returns, cov_matrix, extra_args=()):
    n = len(mean_returns)
    args = (mean_returns, cov_matrix) + extra_args
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))  # long-only portfolio
    init_guess = np.repeat(1 / n, n)

    result = minimize(
        objective_fn, init_guess, args=args,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9}
    )
    return result


def main():
    print("=" * 64)
    print("MEAN-VARIANCE PORTFOLIO OPTIMIZATION")
    print("=" * 64)
    print(f"Assets: {ASSET_NAMES}")
    print(f"Simulated trading days: {N_DAYS}  (seed=42, reproducible)")
    print()

    # --- Equal-weight baseline ---
    equal_w = np.repeat(1 / N_ASSETS, N_ASSETS)
    eq_ret, eq_vol = portfolio_performance(equal_w, mean_returns, cov_matrix)
    eq_sharpe = (eq_ret - RISK_FREE_RATE) / eq_vol

    # --- Minimum-variance portfolio ---
    mv_result = optimize_portfolio(portfolio_variance, mean_returns, cov_matrix)
    mv_weights = mv_result.x
    mv_ret, mv_vol = portfolio_performance(mv_weights, mean_returns, cov_matrix)
    mv_sharpe = (mv_ret - RISK_FREE_RATE) / mv_vol

    # --- Maximum Sharpe ratio portfolio ---
    ms_result = optimize_portfolio(neg_sharpe, mean_returns, cov_matrix,
                                    extra_args=(RISK_FREE_RATE,))
    ms_weights = ms_result.x
    ms_ret, ms_vol = portfolio_performance(ms_weights, mean_returns, cov_matrix)
    ms_sharpe = (ms_ret - RISK_FREE_RATE) / ms_vol

    def report(name, weights, ret, vol, sharpe, converged):
        print(f"--- {name} ---")
        for a, w in zip(ASSET_NAMES, weights):
            print(f"  {a:<14s}: {w*100:6.2f}%")
        print(f"  Expected annual return : {ret*100:6.2f}%")
        print(f"  Annual volatility       : {vol*100:6.2f}%")
        print(f"  Sharpe ratio (rf=4%)    : {sharpe:6.3f}")
        print(f"  Solver converged        : {converged}")
        print()

    report("Equal-Weight Baseline", equal_w, eq_ret, eq_vol, eq_sharpe, "n/a")
    report("Minimum-Variance Portfolio", mv_weights, mv_ret, mv_vol, mv_sharpe, mv_result.success)
    report("Maximum-Sharpe Portfolio", ms_weights, ms_ret, ms_vol, ms_sharpe, ms_result.success)

    improvement = ((ms_sharpe - eq_sharpe) / eq_sharpe) * 100
    print(f"Sharpe ratio improvement vs. equal-weight baseline: {improvement:.1f}%")


if __name__ == "__main__":
    main()
