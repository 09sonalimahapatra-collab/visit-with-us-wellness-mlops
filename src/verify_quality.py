"""A small quality gate used before GitHub Actions commits a new model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .train_model import METRICS_PATH
except ImportError:  # pragma: no cover - exercised by the GitHub Actions command
    from train_model import METRICS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Check post-training model quality thresholds")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--minimum-roc-auc", type=float, default=0.80)
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    failed_checks = []
    if metrics["roc_auc"] < args.minimum_roc_auc:
        failed_checks.append(f"roc_auc={metrics['roc_auc']} < {args.minimum_roc_auc}")
    if metrics["recall"] < args.minimum_recall:
        failed_checks.append(f"recall={metrics['recall']} < {args.minimum_recall}")

    if failed_checks:
        raise RuntimeError("Quality gate failed: " + "; ".join(failed_checks))

    print(
        "QUALITY GATE: PASSED "
        f"(ROC-AUC={metrics['roc_auc']}, recall={metrics['recall']})"
    )


if __name__ == "__main__":
    main()
