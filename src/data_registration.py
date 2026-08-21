"""Validate and register the raw Wellness Tourism customer dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Works both as a module and as ``python src/data_registration.py``.
    from .config import EXPECTED_COLUMNS, RAW_DATA_PATH, REPORTS_DIR, TARGET_COLUMN
except ImportError:  # pragma: no cover - exercised by the GitHub Actions command
    from config import EXPECTED_COLUMNS, RAW_DATA_PATH, REPORTS_DIR, TARGET_COLUMN


def validate_dataset(csv_path: Path) -> dict[str, Any]:
    """Check schema, target values and basic data quality; return a summary."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    unexpected_target_values = sorted(set(frame[TARGET_COLUMN].dropna().unique()) - {0, 1})
    if unexpected_target_values:
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0 and 1; found {unexpected_target_values}"
        )

    if frame.empty:
        raise ValueError("Dataset has no rows.")

    summary: dict[str, Any] = {
        "validation_status": "passed",
        "dataset_path": str(csv_path),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "business_columns_validated": EXPECTED_COLUMNS,
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {column: int(count) for column, count in frame.isna().sum().items()},
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts(dropna=False).sort_index().items()
        },
    }
    return summary


def register_dataset(csv_path: Path, summary_path: Path | None = None) -> dict[str, Any]:
    """Validate the CSV and persist a machine-readable registration summary."""
    summary = validate_dataset(csv_path)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Register and validate tourism.csv")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Path to raw tourism CSV")
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPORTS_DIR / "data_registration_summary.json",
        help="Path for the validation summary JSON",
    )
    args = parser.parse_args()

    summary = register_dataset(args.input, args.summary)
    print("DATA REGISTRATION: PASSED")
    print(f"Rows: {summary['rows']:,} | Columns: {summary['columns']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Target distribution: {summary['target_distribution']}")
    print(f"Validation summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
