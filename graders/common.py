from typing import Any, Dict, Iterable


def clamp(value: float) -> float:
    return max(0.01, min(0.99, round(value, 4)))


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
