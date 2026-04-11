import sys
from pathlib import Path

# Hub may load this file by path; ensure repo root is on sys.path for graders.common.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from graders.common import clamp, extract_metrics, finalize_task_score


def easy_grade(state, action=None, result=None):
    metrics = extract_metrics(state)
    if metrics["service_level"] >= 0.98 and metrics["rejection_free_ratio"] >= 0.98:
        raw = clamp(0.99)
    else:
        raw = clamp(0.6 * metrics["service_level"] + 0.4 * metrics["rejection_free_ratio"])
    return finalize_task_score(raw)


def grade(state, action=None, result=None):
    """Entry point named in openenv.yaml (graders/easy.py)."""
    return easy_grade(state, action, result)
