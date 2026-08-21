"""Clean the raw dataset and create reproducible train/test CSV splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import IDENTIFIER_COLUMNS, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN
    from .data_registration import validate_dataset
except ImportError:  # pragma: no cover - exercised by the GitHub Actions command
    from config import IDENTIFIER_COLUMNS, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN
    from data_registration import validate_dataset


def clean_data(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Standardize known label inconsistencies and remove non-predictive IDs."""
    frame = raw_frame.copy()
    original_rows = len(frame)

    # CustomerID makes each source row unique.  Remove only exact duplicate raw
    # records, never separate customers that happen to share the same profile.
    raw_duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)

    # Whitespace can create different categories even when the text looks alike.
    object_columns = frame.select_dtypes(include="object").columns
    for column in object_columns:
        frame[column] = frame[column].str.strip()

    # The file contains two spellings of the same label.  Combining them avoids
    # treating a data-entry artefact as a meaningful customer segment.
    frame["Gender"] = frame["Gender"].replace({"Fe Male": "Female"})
    # "Unmarried" and "Single" represent the same non-married status for this use case.
    frame["MaritalStatus"] = frame["MaritalStatus"].replace({"Unmarried": "Single"})

    columns_to_drop = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    frame = frame.drop(columns=columns_to_drop)

    cleaning_summary: dict[str, int | list[str]] = {
        "rows_before_cleaning": int(original_rows),
        "rows_after_cleaning": int(len(frame)),
        "duplicates_removed": raw_duplicate_rows,
        "dropped_columns": columns_to_drop,
    }
    return frame, cleaning_summary


def prepare_splits(input_path: Path, output_dir: Path, test_size: float = 0.20) -> dict[str, int | list[str]]:
    """Load raw data, clean it, and persist a stratified train/test split."""
    validate_dataset(input_path)
    cleaned_frame, summary = clean_data(pd.read_csv(input_path))

    features = cleaned_frame.drop(columns=[TARGET_COLUMN])
    target = cleaned_frame[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(output_dir / "y_test.csv", index=False)

    summary.update(
        {
            "test_size": test_size,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_columns": list(features.columns),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
        }
    )
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reproducible tourism train/test splits")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Raw CSV path")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR, help="Split output directory")
    args = parser.parse_args()

    summary = prepare_splits(args.input, args.output_dir)
    print("DATA PREPARATION: PASSED")
    print(f"Dropped columns: {summary['dropped_columns']}")
    print(f"Train rows: {summary['training_rows']:,} | Test rows: {summary['test_rows']:,}")
    print(f"Splits saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
