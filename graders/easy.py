from graders.common import clamp

def easy_grade(metrics):
    # Perfect case (safe float handling)
    if metrics["service_level"] >= 0.999 and metrics["rejection_free_ratio"] >= 0.999:
        return clamp(0.99)

    score = (
        0.6 * metrics["service_level"]
        + 0.4 * metrics["rejection_free_ratio"]
    )

    return clamp(score)