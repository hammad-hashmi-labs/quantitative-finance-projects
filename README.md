# Quantitative Finance Projects

A small collection of self-contained Python projects applying quantitative
and statistical methods, built with `NumPy` and `SciPy`.

## Projects

### [Portfolio Optimization Engine](./portfolio-optimization)
A constrained mean-variance (Markowitz) optimizer that solves for the
minimum-variance and maximum-Sharpe-ratio portfolios using `scipy.optimize`,
then tests them out-of-sample: weights are fit on a chronological training
window and judged on held-out data, repeated across 500 simulated paths. The
result illustrates estimation error in mean-variance optimization: the
max-Sharpe portfolio's median Sharpe ratio fell from 0.908 in-sample to 0.279
out-of-sample, and it beat an equal-weight baseline in 45.6% of paths.

### [Statistical Hypothesis Testing Framework](./hypothesis-testing)
A two-sample inference pipeline that cross-validates statistical
significance through three independent methods: Welch's *t*-test, a
10,000-iteration permutation test, and a bootstrap confidence interval,
rather than relying on a single test's assumptions.

## Notes
Each project folder has its own `README.md` with full methodology, results,
data notes, and run instructions. Data in both projects is simulated
(seeded for reproducibility) to demonstrate the underlying statistical and
optimization techniques, not to represent live market data or a specific
real-world experiment.

## Author
**Hammad Hashmi** — [LinkedIn](https://www.linkedin.com/in/hammad-hashmi) · [GitHub](https://github.com/hammad-hashmi-labs)
