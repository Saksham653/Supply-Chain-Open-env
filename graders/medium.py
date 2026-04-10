"""Medium task grader for multi-SKU inventory balancing."""

from graders.common import clamp, extract_metrics


def grade(state, action=None, result=None):
    metrics = extract_metrics(state)
    score = 0.40 * metrics["service_level"]
    score += 0.20 * metrics["stockout_free_ratio"]
    score += 0.20 * metrics["overstock_free_ratio"]
    score += 0.10 * metrics["rejection_free_ratio"]
    score += 0.10 * metrics["average_reward"]
    return clamp(score)
