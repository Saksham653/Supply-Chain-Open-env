"""Easy task grader for the seeded single-SKU episode."""

from graders.common import clamp, extract_metrics


def grade(state, action=None, result=None):
    metrics = extract_metrics(state)
    if metrics["service_level"] == 1.0 and metrics["rejection_free_ratio"] == 1.0:
        return clamp(1.0)

    score = 0.75 * metrics["service_level"]
    score += 0.15 * metrics["stockout_free_ratio"]
    score += 0.10 * metrics["average_reward"]
    return clamp(score)
