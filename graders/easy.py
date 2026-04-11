from graders.common import clamp, extract_metrics


def easy_grade(state, action=None, result=None):
    metrics = extract_metrics(state)
    if metrics["service_level"] >= 0.98 and metrics["rejection_free_ratio"] >= 0.98:
        return clamp(0.99)
    score = 0.6 * metrics["service_level"] + 0.4 * metrics["rejection_free_ratio"]
    return clamp(score)


# openenv.yaml references graders/easy.py — loaders often expect this name
grade = easy_grade
