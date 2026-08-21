"""Train, tune, evaluate and track the Wellness Tourism purchase model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .config import (
        CATEGORICAL_COLUMNS,
        EXPERIMENTS_DIR,
        MODELS_DIR,
        NUMERIC_COLUMNS,
        PROCESSED_DATA_DIR,
        RANDOM_STATE,
        REPORTS_DIR,
        TARGET_COLUMN,
    )
except ImportError:  # pragma: no cover - exercised by the GitHub Actions command
    from config import (
        CATEGORICAL_COLUMNS,
        EXPERIMENTS_DIR,
        MODELS_DIR,
        NUMERIC_COLUMNS,
        PROCESSED_DATA_DIR,
        RANDOM_STATE,
        REPORTS_DIR,
        TARGET_COLUMN,
    )


MODEL_PATH = MODELS_DIR / "wellness_package_model.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
METADATA_PATH = REPORTS_DIR / "model_metadata.json"
GRID_RESULTS_PATH = EXPERIMENTS_DIR / "grid_search_results.csv"


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read the train/test files downloaded from the GitHub Actions artifact."""
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split artifact files in {data_dir}: {missing}. Run data_preparation.py first."
        )
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")[TARGET_COLUMN].astype(int)
    y_test = pd.read_csv(data_dir / "y_test.csv")[TARGET_COLUMN].astype(int)
    return X_train, X_test, y_train, y_test


def build_model_pipeline() -> Pipeline:
    """Create one pipeline so training and Streamlit inference transform data identically."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def parameter_grid() -> dict[str, list[Any]]:
    """Small, reproducible grid that finishes quickly in GitHub Actions."""
    return {
        "classifier__n_estimators": [200, 350],
        "classifier__max_depth": [None, 12],
        "classifier__min_samples_leaf": [1, 3],
        "classifier__max_features": ["sqrt"],
    }


def calculate_metrics(y_true: pd.Series, probabilities: pd.Series, threshold: float = 0.50) -> dict[str, float | int]:
    """Calculate business-relevant metrics on the untouched test split."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "decision_threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "true_negatives": int(confusion_matrix(y_true, predictions)[0, 0]),
        "false_positives": int(confusion_matrix(y_true, predictions)[0, 1]),
        "false_negatives": int(confusion_matrix(y_true, predictions)[1, 0]),
        "true_positives": int(confusion_matrix(y_true, predictions)[1, 1]),
    }


def save_visual_reports(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: pd.Series,
) -> None:
    """Save the most useful validation and explainability visuals for the notebook."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= 0.50).astype(int)

    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No purchase", "Purchase"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importances = best_model.named_steps["classifier"].feature_importances_
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_frame.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    top_features = importance_frame.head(15).sort_values("importance")
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    axis.barh(top_features["feature"], top_features["importance"], color="#0B6E8A")
    axis.set_xlabel("Random forest feature importance")
    axis.set_title("Top 15 signals used by the model")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def track_with_mlflow(
    best_model: Pipeline,
    best_parameters: dict[str, Any],
    metrics: dict[str, float | int],
) -> str:
    """Log experiment data to local MLflow when available, with a file-log fallback."""
    try:
        import mlflow
        import mlflow.sklearn
    except ModuleNotFoundError:
        return "file-based tracking only (MLflow is not installed in this runtime)"

    tracking_dir = EXPERIMENTS_DIR / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{tracking_dir.resolve()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("wellness-tourism-package-prediction")
    with mlflow.start_run(run_name="random-forest-grid-search"):
        mlflow.log_params({key: str(value) for key, value in best_parameters.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, float)})
        mlflow.log_artifact(str(GRID_RESULTS_PATH), artifact_path="experiments")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.sklearn.log_model(best_model, name="wellness-package-model")
    return f"MLflow tracking URI: {tracking_uri}"


def train_and_evaluate(data_dir: Path) -> dict[str, Any]:
    """Tune the model, evaluate it once on test data, and save model artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_splits(data_dir)

    search = GridSearchCV(
        estimator=build_model_pipeline(),
        param_grid=parameter_grid(),
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        return_train_score=True,
        refit=True,
    )
    search.fit(X_train, y_train)

    grid_results = pd.DataFrame(search.cv_results_)
    grid_results["params"] = grid_results["params"].astype(str)
    selected_columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score",
    ]
    grid_results[selected_columns].sort_values("rank_test_score").to_csv(GRID_RESULTS_PATH, index=False)

    best_model: Pipeline = search.best_estimator_
    probabilities = pd.Series(best_model.predict_proba(X_test)[:, 1], index=X_test.index)
    metrics = calculate_metrics(y_test, probabilities)
    metrics["best_cv_roc_auc"] = round(float(search.best_score_), 4)
    metrics["model_name"] = "RandomForestClassifier"
    metrics["selection_metric"] = "roc_auc"
    metrics["selection_reason"] = "ROC-AUC measures ranking quality across thresholds for an imbalanced target."

    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metadata: dict[str, Any] = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(MODEL_PATH),
        "model_name": "RandomForestClassifier",
        "best_parameters": search.best_params_,
        "cv_folds": 5,
        "cv_scoring": "roc_auc",
        "input_features": list(X_train.columns),
        "categorical_features": CATEGORICAL_COLUMNS,
        "numeric_features": NUMERIC_COLUMNS,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_visual_reports(best_model, X_test, y_test, probabilities)
    tracking_status = track_with_mlflow(best_model, search.best_params_, metrics)
    metadata["tracking_status"] = tracking_status
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = classification_report(y_test, (probabilities >= 0.50).astype(int), digits=4)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    return {"metrics": metrics, "metadata": metadata, "tracking_status": tracking_status}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and track wellness package prediction model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing train/test files from the workflow artifact",
    )
    args = parser.parse_args()
    result = train_and_evaluate(args.data_dir)
    metrics = result["metrics"]
    print("MODEL TRAINING: PASSED")
    print(f"Best CV ROC-AUC: {metrics['best_cv_roc_auc']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f} | Test F1: {metrics['f1_score']:.4f}")
    print(f"Best parameters: {result['metadata']['best_parameters']}")
    print(f"Tracking: {result['tracking_status']}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
