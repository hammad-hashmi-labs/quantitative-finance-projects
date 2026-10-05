# Two-Sample Statistical Hypothesis Testing

A rigorous two-sample comparison pipeline built with `NumPy` and `SciPy.stats`,
cross-validating significance across three independent methods.

## What it does
1. **Welch's t-test** (parametric, unequal-variance)
2. **Permutation test** (10,000 resamples, distribution-free)
3. **Cohen's d** effect size
4. **Bootstrap confidence interval** (10,000 resamples) on the mean difference

Running all three independent approaches side-by-side guards against relying
on a single test's assumptions — if a parametric test, a non-parametric test,
and a resampling-based CI all agree, the result is far more robust than any
one method alone.

## Results (this run, seed=7)
| Metric | Value |
|---|---|
| Mean A (control) | 99.247 |
| Mean B (treatment) | 104.629 |
| Welch's t-statistic | 3.2327 |
| t-test p-value | 0.00136 |
| Cohen's d | 0.373 (small-to-medium effect) |
| Permutation test p-value | 0.00210 |
| 95% Bootstrap CI (mean diff) | [2.095, 8.651] |

All three methods agree: the difference is statistically significant
(p < 0.01) and the 95% CI excludes zero.

## Data note
Group data is **simulated** (seeded for reproducibility) to demonstrate the
statistical methodology end-to-end, not a claim about a specific real-world
dataset or experiment.

## Run it
```bash
pip install numpy scipy
python hypothesis_testing.py
```

## Techniques demonstrated
Parametric hypothesis testing · Permutation/resampling methods · Effect size
calculation · Bootstrap confidence intervals · Reproducible simulation design
