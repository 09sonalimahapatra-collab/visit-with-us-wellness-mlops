"""Shared project configuration and dataset schema."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

RAW_DATA_PATH = DATA_DIR / "tourism.csv"
TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42

# The raw file contains an accidental CSV index called ``Unnamed: 0``.  It is
# allowed at registration time, but it is deliberately removed before fitting.
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

IDENTIFIER_COLUMNS = ["Unnamed: 0", "CustomerID"]

FEATURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column not in {"CustomerID", TARGET_COLUMN}]

CATEGORICAL_COLUMNS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
