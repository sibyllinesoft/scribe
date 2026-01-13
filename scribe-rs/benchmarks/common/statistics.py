"""Statistical helper functions for benchmarks."""

import math


def mean(values: list) -> float:
    """Calculate the arithmetic mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: list) -> float:
    """Calculate the sample standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def confidence_interval_95(values: list) -> tuple[float, float]:
    """Calculate 95% confidence interval using t-distribution approximation.

    Returns:
        Tuple of (lower_bound, upper_bound) for the 95% CI.
    """
    if len(values) < 2:
        m = mean(values)
        return (m, m)

    n = len(values)
    m = mean(values)
    s = std_dev(values)

    # t-values for 95% CI (two-tailed) for various sample sizes
    t_values = {
        2: 12.71,
        3: 4.30,
        4: 3.18,
        5: 2.78,
        6: 2.57,
        7: 2.45,
        8: 2.36,
        9: 2.31,
        10: 2.26,
        15: 2.14,
        20: 2.09,
        30: 2.04,
    }
    t = t_values.get(n, 1.96)  # Fall back to z-score for large n

    margin = t * s / math.sqrt(n)
    return (m - margin, m + margin)


def estimate_tokens(content: str) -> int:
    """Rough token estimation (characters / 4).

    This is a standard approximation for code tokenization.
    """
    return len(content) // 4


def percentage_change(old: float, new: float) -> float:
    """Calculate percentage change from old to new value.

    Returns positive for increase, negative for decrease.
    """
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return ((new - old) / old) * 100
