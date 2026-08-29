# Visit with Us: Wellness Tourism Package Prediction

This project implements a reproducible MLOps workflow to predict whether a customer is likely to purchase a Wellness Tourism Package before marketing outreach.

## Objective

The target variable is `ProdTaken`:

- `1` — customer purchased the package
- `0` — customer did not purchase the package

The solution validates the dataset, prepares reproducible train/test data, compares and tunes classification models, saves the approved model, and automates the workflow using GitHub Actions.

## MLOps Workflow

1. Validate `data/tourism.csv` against the expected schema.
2. Clean the data and create an 80:20 stratified train/test split.
3. Upload the train/test files as a GitHub Actions artifact.
4. Train and tune tree-based classification models using five-fold cross-validation.
5. Select and serialize the approved Random Forest model.
6. Save metrics, feature importance, experiment outputs, and validation reports.
7. Run the workflow automatically on pushes to the `main` branch.

## Model Selection

A class-balanced Random Forest was selected after comparing permitted tree-based models and tuning hyperparameters with `GridSearchCV`.

The model is used as a lead-prioritization tool: it returns a purchase probability and supports marketing decisions rather than making decisions automatically.

## Public Deployment and Evidence

- **GitHub repository:**  
  https://github.com/09sonalimahapatra-collab/visit-with-us-wellness-mlops

- **GitHub Actions workflow:**  
  https://github.com/09sonalimahapatra-collab/visit-with-us-wellness-mlops/actions

- **Public Hugging Face Space:**  
  https://huggingface.co/spaces/09sonali/wellness-tourism-predictor

The public Hugging Face Space uses a Gradio interface to collect customer details, create a one-row pandas DataFrame, load the serialized model, and return a purchase probability with a recommended outreach action.

## Key Repository Structure

```text
visit_with_us_mlops/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── app.py
├── data/
│   ├── tourism.csv
│   └── processed/
├── experiments/
├── models/
│   └── wellness_package_model.joblib
├── reports/
├── src/
│   ├── config.py
│   ├── data_registration.py
│   ├── data_preparation.py
│   ├── generate_eda_assets.py
│   ├── train_model.py
│   └── verify_quality.py
├── tests/
│   └── test_data_registration.py
└── requirements.txt
```

## Run the ML Pipeline Locally

```bash
python -m pip install -r requirements.txt
python src/data_registration.py
python src/data_preparation.py
python src/train_model.py
python src/verify_quality.py
```

## Responsible Use

The predicted probability is intended to prioritize outreach. Marketing decisions should also consider customer consent, communication preferences, campaign policy, and ongoing model-performance monitoring.
