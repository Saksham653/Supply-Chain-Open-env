import math
from typing import Any, Dict, Iterable

# Interior band for metrics; final task scores use finalize_task_score().
_SCORE_LO = 0.001
_SCORE_HI = 0.999


def finalize_task_score(value: float) -> float:
    """Last step before returning from OpenEnv grader: strict (0, 1), never 0.0 or 1.0."""
    if not math.isfinite(value):
        return 0.5
    eps = 1e-5
    v = float(value)
    v = max(eps, min(1.0 - eps, v))
    if v <= 0.0:
        v = eps
    if v >= 1.0:
        v = 1.0 - eps
    return v


def clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    value = float(value)
    if value >= 1.0:
        value = _SCORE_HI
    elif value <= 0.0:
        value = _SCORE_LO
    else:
        value = round(value, 6)
        if value >= 1.0:
            value = _SCORE_HI
        elif value <= 0.0:
            value = _SCORE_LO
    value = max(_SCORE_LO, min(_SCORE_HI, value))
    if value <= 0.0 or value >= 1.0:
        return 0.5
    return float(value)


def safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def average(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return sum(values) / len(values)


def extract_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    trajectory = state.get("trajectory", [])
    max_days = max(1, int(state.get("max_days", len(trajectory) or 1)))
    service_level = float(state.get("service_level", 0.0))
    average_reward = float(
        state.get("average_reward", average((step.get("reward", 0.0) for step in trajectory), 0.0))
    )
    rejection_rate = float(state.get("rejection_rate", 0.0))
    stockout_days = float(state.get("stockout_days", 0.0))
    overstock_days = float(state.get("overstock_days", 0.0))
    critical_days = float(state.get("critical_days", 0.0))

    half_index = len(trajectory) // 2
    late_rewards = [float(step.get("reward", 0.0)) for step in trajectory[half_index:]]

    return {
        "service_level": clamp(service_level),
        "average_reward": clamp(average_reward),
        "stockout_free_ratio": clamp(1.0 - safe_ratio(stockout_days, max_days, 0.0)),
        "overstock_free_ratio": clamp(1.0 - safe_ratio(overstock_days, max_days, 0.0)),
        "critical_free_ratio": clamp(1.0 - safe_ratio(critical_days, max_days, 0.0)),
        "rejection_free_ratio": clamp(1.0 - rejection_rate),
        "late_reward_average": clamp(average(late_rewards, average_reward)),
    }
