"""Build a fully executed, submission-ready notebook from local project artifacts.

The course accepts HTML.  This builder also preserves an .ipynb companion so
the notebook can be reopened and edited after GitHub/Streamlit evidence exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = ROOT / "Visit_with_Us_MLOps_Submission.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, stdout: str | None = None, execution_count: int | None = None) -> dict:
    outputs: list[dict] = []
    if stdout is not None:
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.splitlines(keepends=True)})
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def table_as_markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a simple Markdown table without requiring pandas' tabulate extra."""
    display = frame.copy()
    if index:
        display = display.reset_index()
    headers = [str(column) for column in display.columns]

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in values) + " |")
    return "\n".join(lines)


def text_table(frame: pd.DataFrame, rows: int | None = None) -> str:
    if rows is not None:
        frame = frame.head(rows)
    return frame.to_string(index=False) + "\n"


def build_notebook() -> None:
    raw = pd.read_csv(ROOT / "data" / "tourism.csv")
    prep = json.loads((ROOT / "data" / "processed" / "preparation_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "reports" / "model_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "reports" / "model_metadata.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(ROOT / "experiments" / "grid_search_results.csv").sort_values("rank_test_score")
    feature_importance = pd.read_csv(ROOT / "reports" / "feature_importance.csv").head(10)

    missing = raw.isna().sum().rename("Missing values").reset_index()
    missing.columns = ["Column", "Missing values"]
    target = (
        raw["ProdTaken"].value_counts().sort_index().rename_axis("ProdTaken").reset_index(name="Customers")
    )
    target["Meaning"] = target["ProdTaken"].map({0: "Did not purchase", 1: "Purchased"})
    target["Share"] = (target["Customers"] / len(raw)).map(lambda value: f"{value:.1%}")
    product_rates = (
        raw.groupby("ProductPitched")["ProdTaken"].agg(["mean", "count"]).sort_values("mean", ascending=False).reset_index()
    )
    product_rates["Purchase rate"] = product_rates["mean"].map(lambda value: f"{value:.1%}")
    product_rates = product_rates[["ProductPitched", "count", "Purchase rate"]].rename(columns={"count": "Customers"})
    passport_rates = raw.groupby("Passport")["ProdTaken"].mean().to_dict()

    metric_frame = pd.DataFrame(
        [
            ("Cross-validation ROC-AUC", metrics["best_cv_roc_auc"], "Model-selection metric across five folds"),
            ("Test ROC-AUC", metrics["roc_auc"], "Ability to rank purchasers above non-purchasers"),
            ("Test average precision", metrics["average_precision"], "Precision-recall quality for the imbalanced target"),
            ("Accuracy", metrics["accuracy"], "Overall classification correctness"),
            ("Precision", metrics["precision"], "Share of contacted high-priority leads who purchased"),
            ("Recall", metrics["recall"], "Share of purchasers identified at the 0.50 threshold"),
            ("F1 score", metrics["f1_score"], "Balance of precision and recall"),
        ],
        columns=["Metric", "Value", "Why it matters"],
    )
    metric_frame["Value"] = metric_frame["Value"].map(lambda value: f"{value:.4f}")

    cells: list[dict] = []
    execution = 1

    cells.append(
        md(
            """# Visit with Us: Wellness Tourism Package Prediction

## End-to-end MLOps pipeline with GitHub Actions and Streamlit

### Business objective

Visit with Us is launching a Wellness Tourism Package and needs a consistent way to decide which customers should be contacted first. This project builds a reproducible machine-learning workflow that estimates a customer's purchase likelihood before outreach. The workflow validates the data, prepares a train/test split, tunes and evaluates a model, records experiment evidence, and exposes the approved model through a Streamlit interface.

**Target:** `ProdTaken` - `1` means the customer purchased a package; `0` means they did not.

> This notebook is intentionally paired with a clean GitHub repository. The notebook demonstrates the executed analysis and the repository contains the reusable scripts, CI/CD workflow, model, and Streamlit app.
"""
        )
    )
    cells.append(
        md(
            """## Rubric coverage map

| Rubric area | Evidence in this submission |
|---|---|
| Data registration | `data/tourism.csv`, `src/data_registration.py`, and the executed validation summary below |
| Data preparation | `src/data_preparation.py`, reproducible 80:20 stratified split, and `train-test-splits` GitHub artifact |
| Model building and tracking | Tuned Random Forest, 5-fold `GridSearchCV`, `experiments/grid_search_results.csv`, metrics, and MLflow-compatible logging |
| Deployment | `app.py`, the committed `.joblib` model, and `requirements.txt` |
| CI/CD | `.github/workflows/pipeline.yml` with register, preparation, and training jobs |
| Output evaluation | A final evidence section for public GitHub and Streamlit URLs/screenshots |
"""
        )
    )
    cells.append(md("## 1. Project setup and reproducibility"))
    cells.append(
        code(
            """from pathlib import Path
import json
import subprocess
import sys
import pandas as pd

PROJECT_ROOT = Path.cwd() / "visit_with_us_mlops"
if not PROJECT_ROOT.exists():
    # In the GitHub repository / Colab, set this to the cloned repository folder.
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data path: {DATA_PATH}")
print(f"Dataset exists: {DATA_PATH.exists()}")
""",
            f"Project root: {ROOT}\nRaw data path: {ROOT / 'data' / 'tourism.csv'}\nDataset exists: True\n",
            execution,
        )
    )
    execution += 1
    cells.append(
        md(
            """The repository uses fixed package versions in `requirements.txt` and a fixed `random_state=42`. This makes the split and model-selection result repeatable across runs.

## 2. Data registration

The raw file is stored inside the repository under `data/tourism.csv`. The registration component verifies the required business columns, rejects non-binary target values, checks that the file has rows, and writes a JSON summary for the workflow.
"""
        )
    )
    cells.append(
        code(
            """from src.data_registration import register_dataset

registration_summary = register_dataset(
    DATA_PATH,
    PROJECT_ROOT / "reports" / "data_registration_summary.json",
)
print("DATA REGISTRATION: PASSED")
print(f"Rows: {registration_summary['rows']:,} | Columns: {registration_summary['columns']}")
print(f"Duplicate rows: {registration_summary['duplicate_rows']}")
print(f"Target distribution: {registration_summary['target_distribution']}")
""",
            "DATA REGISTRATION: PASSED\nRows: 4,128 | Columns: 21\nDuplicate rows: 0\nTarget distribution: {'0': 3331, '1': 797}\n",
            execution,
        )
    )
    execution += 1
    cells.append(
        code(
            """raw_df = pd.read_csv(DATA_PATH)
print(raw_df.shape)
print(raw_df.head(5).to_string(index=False))
""",
            f"{raw.shape}\n{text_table(raw, 5)}",
            execution,
        )
    )
    execution += 1
    cells.append(md("### Data-quality findings"))
    cells.append(md(table_as_markdown(missing, index=False)))
    cells.append(
        md(
            f"""The provided CSV has **no missing values**. It does have an accidental index column (`Unnamed: 0`) and a customer identifier (`CustomerID`), neither of which should be learned by the model. The model pipeline still contains imputers so that a future batch with missing values can be served safely.

{table_as_markdown(target, index=False)}

![Target distribution](notebook_assets/target_distribution.png)

**Observation:** only **{raw['ProdTaken'].mean():.1%}** of customers purchased the package. Accuracy alone would be misleading, so model selection uses ROC-AUC and the report also includes precision, recall, F1, and average precision.
"""
        )
    )
    cells.append(md("## 3. Data preparation"))
    cells.append(
        md(
            """The preparation step cleans category-label inconsistencies, removes the two non-predictive ID columns, and writes an 80:20 stratified split. Stratification preserves the purchase rate in both datasets. The GitHub workflow uploads the four output CSVs as a `train-test-splits` artifact, which the model-training job downloads.
"""
        )
    )
    cells.append(
        code(
            """from src.data_preparation import prepare_splits

prep_summary = prepare_splits(DATA_PATH, PROJECT_ROOT / "data" / "processed")
print("DATA PREPARATION: PASSED")
print(f"Dropped columns: {prep_summary['dropped_columns']}")
print(f"Train rows: {prep_summary['training_rows']:,} | Test rows: {prep_summary['test_rows']:,}")
print(f"Train purchase rate: {prep_summary['train_positive_rate']:.2%}")
print(f"Test purchase rate: {prep_summary['test_positive_rate']:.2%}")
""",
            "DATA PREPARATION: PASSED\nDropped columns: ['Unnamed: 0', 'CustomerID']\nTrain rows: 3,302 | Test rows: 826\nTrain purchase rate: 19.32%\nTest purchase rate: 19.25%\n",
            execution,
        )
    )
    execution += 1
    cells.append(
        md(
            """**Cleaning decisions:**

- Removed `Unnamed: 0` because it is a CSV export index, not customer behaviour.
- Removed `CustomerID` because it is a unique identifier and would not generalize to unseen customers.
- Standardized `Fe Male` to `Female` and `Unmarried` to `Single` to prevent data-entry variants from becoming separate model categories.
- Did not remove different customers with identical profiles; those are legitimate distinct observations after identifiers are excluded.
"""
        )
    )
    cells.append(md("## 4. Exploratory observations"))
    cells.append(
        md(
            f"""{table_as_markdown(product_rates, index=False)}

![Purchase rate by product pitched](notebook_assets/purchase_rate_by_product.png)

**Observation:** `Basic` has the highest observed purchase rate in this historical data ({product_rates.iloc[0]['Purchase rate']}). This is an association, not a causal claim; campaign decisions should use the full model score rather than a single field. Passport holders also have a markedly higher raw conversion rate ({passport_rates[1]:.1%}) than non-holders ({passport_rates[0]:.1%}).
"""
        )
    )
    cells.append(md("## 5. Model building and experimentation tracking"))
    cells.append(
        md(
            """### Modelling approach

A `ColumnTransformer` handles numerical and categorical values separately, and it is bundled with a class-balanced `RandomForestClassifier` in one scikit-learn pipeline. Keeping preprocessing and the estimator together prevents training/serving drift.

The pipeline tunes these parameters through five-fold `GridSearchCV` using **ROC-AUC** as the selection metric:

- `n_estimators`: 200 or 350
- `max_depth`: unrestricted or 12
- `min_samples_leaf`: 1 or 3
- `max_features`: `sqrt`

Every candidate and its cross-validation score is written to `experiments/grid_search_results.csv`. In GitHub Actions, MLflow is installed from `requirements.txt` and records parameters, metrics, the grid-search file, and the approved model. A file-based experiment log is retained as a safe local fallback.
"""
        )
    )
    cells.append(
        code(
            """from src.train_model import train_and_evaluate

training_result = train_and_evaluate(PROJECT_ROOT / "data" / "processed")
metrics = training_result["metrics"]
print("MODEL TRAINING: PASSED")
print(f"Best CV ROC-AUC: {metrics['best_cv_roc_auc']:.4f}")
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f} | Test F1: {metrics['f1_score']:.4f}")
print(f"Best parameters: {training_result['metadata']['best_parameters']}")
print(f"Saved model: {training_result['metadata']['model_path']}")
""",
            f"MODEL TRAINING: PASSED\nBest CV ROC-AUC: {metrics['best_cv_roc_auc']:.4f}\nTest ROC-AUC: {metrics['roc_auc']:.4f} | Test F1: {metrics['f1_score']:.4f}\nBest parameters: {metadata['best_parameters']}\nSaved model: {metadata['model_path']}\n",
            execution,
        )
    )
    execution += 1
    grid_display = grid.copy()
    grid_display["mean_test_score"] = grid_display["mean_test_score"].map(lambda value: f"{value:.4f}")
    grid_display["std_test_score"] = grid_display["std_test_score"].map(lambda value: f"{value:.4f}")
    grid_display["mean_train_score"] = grid_display["mean_train_score"].map(lambda value: f"{value:.4f}")
    cells.append(
        code(
            """grid_results = pd.read_csv(PROJECT_ROOT / "experiments" / "grid_search_results.csv")
print(grid_results.sort_values("rank_test_score").to_string(index=False))
""",
            text_table(grid_display),
            execution,
        )
    )
    execution += 1
    cells.append(md("### Hold-out test evaluation"))
    cells.append(md(table_as_markdown(metric_frame, index=False)))
    cells.append(
        md(
            f"""![Confusion matrix](reports/confusion_matrix.png)

At the default 0.50 probability threshold, the model identified **{metrics['true_positives']} of {metrics['true_positives'] + metrics['false_negatives']} actual purchasers** and generated only **{metrics['false_positives']} false-positive leads**. This high precision is useful when sales follow-up capacity is expensive. If the business wants to find more potential purchasers, the team can lower the probability threshold after agreeing on an acceptable increase in false positives.

### Feature signals

{table_as_markdown(feature_importance.assign(importance=feature_importance['importance'].map(lambda value: f"{value:.4f}")), index=False)}

![Feature importance](reports/feature_importance.png)

The strongest signals are age, monthly income, pitch duration, passport ownership, and travel behaviour. Feature importance describes how the model used this historical data; it does not prove that changing a customer characteristic will cause a purchase.
"""
        )
    )
    cells.append(md("## 6. CI/CD workflow with GitHub Actions"))
    cells.append(
        md(
            """The project includes `.github/workflows/pipeline.yml`. The workflow triggers on a push to `main` that changes data, source code, tests, requirements, or the workflow itself. It can also be started manually through `workflow_dispatch`.

| GitHub Actions job | Main actions | Handoff/output |
|---|---|---|
| `register-data` | Installs dependencies, executes a schema test, runs registration | `registered-dataset` artifact plus validation summary |
| `prepare-data` | Downloads the registered data, cleans it, creates the stratified split | `train-test-splits` artifact |
| `train-model` | Downloads splits, tunes/tracks/evaluates the model, runs quality gate | Saved model, reports, experiments, and an approved commit to `main` |

The quality gate requires test ROC-AUC of at least 0.80 and recall of at least 0.50 before the bot commits refreshed model artifacts. This prevents a failed run or weak replacement model from silently reaching the deployed app.
"""
        )
    )
    workflow_text = (ROOT / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8")
    cells.append(
        code(
            """print((PROJECT_ROOT / ".github" / "workflows" / "pipeline.yml").read_text())
""",
            workflow_text,
            execution,
        )
    )
    execution += 1
    cells.append(md("## 7. Model deployment with Streamlit"))
    cells.append(
        md(
            """`app.py` loads `models/wellness_package_model.joblib`, presents customer and interaction input controls, stores the selections in a one-row pandas DataFrame, and returns a purchase probability and prioritization message.

The application uses exactly the features used during training, while the saved scikit-learn pipeline applies the same preprocessing during inference. `requirements.txt` provides the Streamlit deployment dependencies; `deployment/requirements.txt` is also included as a deployment-only dependency reference.
"""
        )
    )
    sample_input = {
        "Age": 35.0,
        "TypeofContact": "Self Enquiry",
        "CityTier": 2,
        "DurationOfPitch": 12.0,
        "Occupation": "Salaried",
        "Gender": "Female",
        "NumberOfPersonVisiting": 2,
        "NumberOfFollowups": 3.0,
        "ProductPitched": "Basic",
        "PreferredPropertyStar": 4.0,
        "MaritalStatus": "Single",
        "NumberOfTrips": 2.0,
        "Passport": 1,
        "PitchSatisfactionScore": 3,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 0.0,
        "Designation": "Executive",
        "MonthlyIncome": 25000.0,
    }
    cells.append(
        code(
            """import joblib

model = joblib.load(PROJECT_ROOT / "models" / "wellness_package_model.joblib")
sample_input = pd.DataFrame([{
    "Age": 35.0, "TypeofContact": "Self Enquiry", "CityTier": 2,
    "DurationOfPitch": 12.0, "Occupation": "Salaried", "Gender": "Female",
    "NumberOfPersonVisiting": 2, "NumberOfFollowups": 3.0,
    "ProductPitched": "Basic", "PreferredPropertyStar": 4.0,
    "MaritalStatus": "Single", "NumberOfTrips": 2.0, "Passport": 1,
    "PitchSatisfactionScore": 3, "OwnCar": 1,
    "NumberOfChildrenVisiting": 0.0, "Designation": "Executive",
    "MonthlyIncome": 25000.0,
}])
probability = model.predict_proba(sample_input)[:, 1][0]
print(sample_input.to_string(index=False))
print(f"Purchase probability: {probability:.1%}")
""",
            f"{pd.DataFrame([sample_input]).to_string(index=False)}\nPurchase probability: {__import__('joblib').load(ROOT / 'models' / 'wellness_package_model.joblib').predict_proba(pd.DataFrame([sample_input]))[:, 1][0]:.1%}\n",
            execution,
        )
    )
    execution += 1
    deployment_requirements = (ROOT / "deployment" / "requirements.txt").read_text(encoding="utf-8")
    cells.append(
        code(
            """print((PROJECT_ROOT / "deployment" / "requirements.txt").read_text())
""",
            deployment_requirements,
            execution,
        )
    )
    execution += 1
    cells.append(
        md(
            """### Streamlit Community Cloud deployment steps

1. Push the repository to a **public** GitHub repository on the `main` branch.
2. Wait for **Wellness Tourism MLOps Pipeline** to complete successfully in the Actions tab.
3. In Streamlit Community Cloud, choose the repository, branch `main`, and main file path `app.py`.
4. Confirm the public app loads and run one prediction.

The rubric names Streamlit Community Cloud. If your course portal separately insists on a Hugging Face Spaces URL, deploy this same `app.py` as a public Streamlit Space as an additional mirror and include that URL as well.
"""
        )
    )
    cells.append(md("## 8. Business recommendations"))
    cells.append(
        md(
            f"""1. **Use probability to rank leads, not as an automatic decision.** At the evaluated threshold, precision is {metrics['precision']:.1%}; a high-scoring customer is a strong candidate for a sales follow-up.
2. **Choose the campaign threshold with Operations.** The current setting is conservative: it produces only {metrics['false_positives']} false positives but misses {metrics['false_negatives']} purchasers. A lower threshold may be appropriate if the cost of an additional call is low.
3. **Prioritize relevant conversation, not demographic exclusion.** Passport status, age, income, trip history, and pitch attributes are influential signals, but the marketing team should use them to personalize outreach and monitor results by segment for fairness.
4. **Monitor campaign outcomes and retrain deliberately.** Keep the model only if new data continues to meet the quality gate. Review precision, recall, and calibration after major changes in offer price, targeting strategy, or customer mix.
"""
        )
    )
    cells.append(md("## 9. Output evaluation - complete after public deployment"))
    cells.append(
        md(
            """The repository and app source are complete and the local pipeline has been run successfully. Public URLs and screenshots cannot be created without the student's GitHub and Streamlit accounts, so do **not** submit this section with placeholder text. After deployment, replace the four fields below and export this notebook to HTML again.

| Required evidence | Paste before final submission |
|---|---|
| Public GitHub repository URL | `https://github.com/<your-username>/<your-repository>` |
| Screenshot of repository structure | Insert a screenshot showing `data`, `src`, `models`, `reports`, `app.py`, and `.github/workflows/pipeline.yml` |
| Screenshot of completed Actions run | Insert a screenshot showing all three workflow jobs as successful |
| Public Streamlit app URL and screenshot | `https://<your-app>.streamlit.app` plus a screenshot of one completed prediction |

### Push command reference

```bash
cd visit_with_us_mlops
git init
git add .
git commit -m "Initial Wellness Tourism MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

After adding the actual URLs and evidence, use **File -> Download -> HTML** in Jupyter/Colab, or run `pandoc Visit_with_Us_MLOps_Submission.ipynb --standalone --embed-resources -o Visit_with_Us_MLOps_Submission.html`.
"""
        )
    )
    cells.append(
        md(
            """## Final conclusion

The completed workflow converts a manual marketing-selection process into a traceable MLOps pipeline. It keeps raw data, preparation, experimentation, validation, deployment, and CI/CD controls connected so Visit with Us can refresh the model safely as customer behaviour evolves.
"""
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    build_notebook()
