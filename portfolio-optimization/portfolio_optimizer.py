"""
Mean-Variance Portfolio Optimization (Markowitz) with Out-of-Sample Evaluation
-------------------------------------------------------------------------------
Computes the Minimum-Variance and Maximum-Sharpe-Ratio portfolios using
constrained optimization (scipy.optimize.minimize, SLSQP), then evaluates
them the way a portfolio would actually be used: weights are FIT on a
training window and JUDGED on later, unseen (out-of-sample) returns.

What this script reports
  1. Single-path run (seed=42): in-sample vs. out-of-sample Sharpe ratios for
     an equal-weight baseline, the minimum-variance portfolio, and the
     maximum-Sharpe portfolio, using a chronological 70/30 train/test split.
  2. Robustness check: the same fit-then-test procedure repeated over many
     independently simulated return paths, because a single path can flatter
     or punish a strategy by luck.

NOTE ON DATA: All returns are SIMULATED (seeded for reproducibility) from a
multivariate normal distribution with asset-specific mean/volatility
assumptions and a specified correlation matrix. This is a methodology
demonstration of estimation error in portfolio optimization, not a claim
about real historical market performance or a tradable strategy.
"""

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ASSET_NAMES = ["Equity_A", "Equity_B", "Bond_A", "Commodity_A", "Equity_C"]
N_ASSETS = len(ASSET_NAMES)
N_DAYS = 1000            # ~4 trading years of daily returns per simulated path
TRAIN_FRAC = 0.70        # first 70% of days = training window, last 30% = test
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
PRIMARY_SEED = 42        # the single reported path
N_SIMS = 500             # Monte Carlo robustness paths
MC_BASE_SEED = 1000      # seeds 1000..1499, disjoint from the primary seed

# Annualized assumptions (mean return, volatility) per asset
ANNUAL_MU = np.array([0.10, 0.12, 0.04, 0.07, 0.14])
ANNUAL_SIGMA = np.array([0.18, 0.22, 0.05, 0.20, 0.28])

# Correlation matrix (symmetric, positive semi-definite by construction)
CORR = np.array([
    [1.00, 0.65, 0.10, 0.20, 0.55],
    [0.65, 1.00, 0.05, 0.15, 0.60],
    [0.10, 0.05, 1.00, 0.00, 0.08],
    [0.20, 0.15, 0.00, 1.00, 0.18],
    [0.55, 0.60, 0.08, 0.18, 1.00],
])

STRATEGIES = ["Equal-Weight", "Min-Variance", "Max-Sharpe"]


# ---------------------------------------------------------------------------
# Data simulation and splitting
# ---------------------------------------------------------------------------
def simulate_returns(seed, n_days=N_DAYS):
    """Simulate daily returns for all assets (reproducible for a given seed)."""
    rng = np.random.RandomState(seed)
    daily_mu = ANNUAL_MU / TRADING_DAYS
    daily_sigma = ANNUAL_SIGMA / np.sqrt(TRADING_DAYS)
    cov_daily = np.outer(daily_sigma, daily_sigma) * CORR
    return rng.multivariate_normal(daily_mu, cov_daily, size=n_days)


def chronological_split(returns, train_frac=TRAIN_FRAC):
    """Split by time (never shuffled): earlier days train, later days test."""
    cut = int(len(returns) * train_frac)
    return returns[:cut], returns[cut:]


# ---------------------------------------------------------------------------
# Estimation and optimization
# ---------------------------------------------------------------------------
def estimate_moments(returns):
    """Annualized mean vector and covariance matrix from daily returns."""
    mean_returns = returns.mean(axis=0) * TRADING_DAYS
    cov_matrix = np.cov(returns, rowvar=False) * TRADING_DAYS
    return mean_returns, cov_matrix


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
    """Long-only, fully invested (weights >= 0, sum to 1) via SLSQP."""
    n = len(mean_returns)
    args = (mean_returns, cov_matrix) + extra_args
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init_guess = np.repeat(1 / n, n)
    return minimize(
        objective_fn, init_guess, args=args,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )


def fit_strategies(train_returns):
    """Fit all three strategies using ONLY the training window.
    Returns ({name: weights}, all_converged)."""
    mean_returns, cov_matrix = estimate_moments(train_returns)
    equal_w = np.repeat(1 / N_ASSETS, N_ASSETS)
    mv = optimize_portfolio(portfolio_variance, mean_returns, cov_matrix)
    ms = optimize_portfolio(neg_sharpe, mean_returns, cov_matrix,
                            extra_args=(RISK_FREE_RATE,))
    weights = {"Equal-Weight": equal_w, "Min-Variance": mv.x, "Max-Sharpe": ms.x}
    return weights, bool(mv.success and ms.success)


# ---------------------------------------------------------------------------
# Evaluation: realized performance of FIXED weights on a window of returns
# ---------------------------------------------------------------------------
def realized_performance(weights, returns):
    """Annualized return, volatility and Sharpe of a fixed-weight portfolio
    over the given window of daily returns."""
    daily = returns @ weights
    ann_ret = daily.mean() * TRADING_DAYS
    ann_vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    return ann_ret, ann_vol, sharpe


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def run_single_path(seed=PRIMARY_SEED):
    returns = simulate_returns(seed)
    train, test = chronological_split(returns)
    weights, converged = fit_strategies(train)

    print("=" * 72)
    print("SINGLE PATH: fit on training window, judged on unseen test window")
    print("=" * 72)
    print(f"Assets: {ASSET_NAMES}")
    print(f"Simulated days: {len(returns)}  (train={len(train)}, test={len(test)}, "
          f"seed={seed}, chronological split)")
    print(f"Solver converged on all fits: {converged}")
    print()

    for name in STRATEGIES:
        print(f"--- {name} weights (fit on training data only) ---")
        for a, w in zip(ASSET_NAMES, weights[name]):
            print(f"  {a:<14s}: {w * 100:6.2f}%")
        print()

    header = f"{'Strategy':<14s} | {'In-sample Sharpe':>16s} | {'Out-of-sample Sharpe':>20s} | {'OOS return':>10s} | {'OOS vol':>8s}"
    print(header)
    print("-" * len(header))
    results = {}
    for name in STRATEGIES:
        _, _, sh_in = realized_performance(weights[name], train)
        r_out, v_out, sh_out = realized_performance(weights[name], test)
        results[name] = (sh_in, sh_out)
        print(f"{name:<14s} | {sh_in:>16.3f} | {sh_out:>20.3f} | {r_out * 100:>9.2f}% | {v_out * 100:>7.2f}%")
    print()

    ms_in, ms_out = results["Max-Sharpe"]
    ew_out = results["Equal-Weight"][1]
    print(f"Max-Sharpe: in-sample {ms_in:.3f} -> out-of-sample {ms_out:.3f} "
          f"(gap {ms_in - ms_out:+.3f})")
    print(f"Max-Sharpe minus Equal-Weight, out-of-sample: {ms_out - ew_out:+.3f} Sharpe")
    print()


def run_monte_carlo(n_sims=N_SIMS, base_seed=MC_BASE_SEED):
    in_sh = {s: [] for s in STRATEGIES}
    out_sh = {s: [] for s in STRATEGIES}
    failures = 0

    for i in range(n_sims):
        returns = simulate_returns(base_seed + i)
        train, test = chronological_split(returns)
        weights, converged = fit_strategies(train)
        if not converged:
            failures += 1
        for name in STRATEGIES:
            in_sh[name].append(realized_performance(weights[name], train)[2])
            out_sh[name].append(realized_performance(weights[name], test)[2])

    in_sh = {k: np.array(v) for k, v in in_sh.items()}
    out_sh = {k: np.array(v) for k, v in out_sh.items()}

    print("=" * 72)
    print(f"ROBUSTNESS: {n_sims} independent simulated paths (seeds {base_seed}-{base_seed + n_sims - 1})")
    print("=" * 72)
    print(f"Paths where the solver did not fully converge: {failures}")
    print()
    header = f"{'Strategy':<14s} | {'Median in-sample':>16s} | {'Median out-of-sample':>20s}"
    print(header)
    print("-" * len(header))
    for name in STRATEGIES:
        print(f"{name:<14s} | {np.median(in_sh[name]):>16.3f} | {np.median(out_sh[name]):>20.3f}")
    print()

    diff = out_sh["Max-Sharpe"] - out_sh["Equal-Weight"]
    win_rate = (diff > 0).mean() * 100
    gap = in_sh["Max-Sharpe"] - out_sh["Max-Sharpe"]
    print(f"Max-Sharpe beat Equal-Weight out-of-sample in {win_rate:.1f}% of paths")
    print(f"Median out-of-sample Sharpe difference (Max-Sharpe - Equal-Weight): {np.median(diff):+.3f}")
    print(f"Median in-sample-to-out-of-sample Sharpe decay for Max-Sharpe: {np.median(gap):+.3f}")
    print()


def main():
    run_single_path()
    run_monte_carlo()


if __name__ == "__main__":
    main()
