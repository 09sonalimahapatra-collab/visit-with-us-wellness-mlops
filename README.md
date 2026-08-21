# Visit with Us - Wellness Tourism Package Prediction

This repository implements a reproducible MLOps workflow that identifies customers who are likely to purchase the Wellness Tourism Package before a marketing call.

## What the pipeline does

1. Validates `data/tourism.csv` against the expected business schema.
2. Cleans the data, removes identifiers, and creates a reproducible 80:20 stratified split.
3. Transfers the train/test files from one GitHub Actions job to the next as an artifact.
4. Tunes a class-balanced Random Forest with 5-fold cross-validation and logs all candidate parameters and scores.
5. Saves the approved model, metrics, feature importance, and experiment log.
6. Commits the refreshed model to `main` only after the validation and quality checks pass.
7. Serves the committed model through `app.py` on Streamlit Community Cloud.

## Repository structure

```text
visit_with_us_mlops/
├── .github/workflows/pipeline.yml
├── app.py
├── data/
│   ├── tourism.csv
│   └── processed/                 # GitHub Actions artifact
├── experiments/                   # Grid-search and MLflow tracking outputs
├── models/wellness_package_model.joblib
├── reports/                       # Registration, metrics and diagnostic outputs
├── src/
│   ├── data_registration.py
│   ├── data_preparation.py
│   ├── generate_eda_assets.py
│   ├── train_model.py
│   └── verify_quality.py
├── tests/test_data_registration.py
└── requirements.txt
```

## Run locally

```bash
python -m pip install -r requirements.txt
python src/data_registration.py
python src/data_preparation.py
python src/train_model.py
python src/verify_quality.py
streamlit run app.py
```

## Deploy for assessment

1. Create a new public GitHub repository, then push this entire folder to its `main` branch.
2. Open **Actions** and confirm that **Wellness Tourism MLOps Pipeline** completes successfully.
3. Create a Streamlit Community Cloud app from the same public repository with **Main file path** set to `app.py`.
4. Paste the two public URLs and insert screenshots of the finished workflow and app into the **Output Evaluation** section of the submission notebook.

The `models/`, `reports/`, and `experiments/` folders are intentionally committed after a successful run so Streamlit always serves the latest approved model.
