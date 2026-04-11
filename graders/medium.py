"""Medium task grader for multi-SKU inventory balancing."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from graders.common import clamp, extract_metrics, finalize_task_score


def medium_grade(state, action=None, result=None):
    metrics = extract_metrics(state)
    score = 0.40 * metrics["service_level"]
    score += 0.20 * metrics["stockout_free_ratio"]
    score += 0.20 * metrics["overstock_free_ratio"]
    score += 0.10 * metrics["rejection_free_ratio"]
    score += 0.10 * metrics["average_reward"]
    return finalize_task_score(clamp(score))


def grade(state, action=None, result=None):
    return medium_grade(state, action, result)
