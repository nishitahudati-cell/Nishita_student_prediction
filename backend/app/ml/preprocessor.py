"""Preprocessing pipelines and feature transformations."""
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from backend.app.core.config import PASS_THRESHOLD, RAW_DATA_DIR
from backend.app.ml.dataset_schema import (
    UCI_CATEGORICAL_FEATURES,
    UCI_NUMERIC_FEATURES_EARLY,
    UCI_NUMERIC_FEATURES_LATE,
    TARGET_COLUMN,
)


def build_preprocessor(
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for categorical and numeric features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def detect_separator(file_path: Path) -> str:
    """Detect whether a CSV file uses semicolon or comma separation."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if ";" in first_line and first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def load_dataset(
    file_path: Optional[Path] = None,
    timing_scenario: str = "early",
    pass_threshold: float = PASS_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Load and prepare feature matrix X and binary target vector y.
    
    Leakage control:
    - Target G3 is stripped from predictors.
    - If timing_scenario == 'early', G1 and G2 are excluded.
    - If timing_scenario == 'late', G1 and G2 are retained.
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "uci_student_performance" / "student-mat.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    sep = detect_separator(file_path)
    df = pd.read_csv(file_path, sep=sep)

    # Determine target column
    if "outcome" in df.columns:
        y = (df["outcome"].astype(str).str.strip().str.lower() == "pass").astype(int)
    elif TARGET_COLUMN in df.columns:
        y = (df[TARGET_COLUMN] >= pass_threshold).astype(int)
    elif "final_grade" in df.columns:
        thresh = 40.0 if df["final_grade"].max() > 20 else pass_threshold
        y = (df["final_grade"] >= thresh).astype(int)
    else:
        raise ValueError(f"Could not find valid target column ({TARGET_COLUMN}, final_grade, or outcome) in dataset.")

    # Drop target and any outcome labels
    drop_cols = [col for col in [TARGET_COLUMN, "final_grade", "outcome", "data_label", "record_id"] if col in df.columns]
    
    # Timing scenario feature gating
    if timing_scenario == "early":
        for col in ["G1", "G2"]:
            if col in df.columns:
                drop_cols.append(col)
        cat_cols = [c for c in UCI_CATEGORICAL_FEATURES if c in df.columns]
        num_cols = [c for c in UCI_NUMERIC_FEATURES_EARLY if c in df.columns]
    else:
        cat_cols = [c for c in UCI_CATEGORICAL_FEATURES if c in df.columns]
        num_cols = [c for c in UCI_NUMERIC_FEATURES_LATE if c in df.columns]

    # If dataset has generic columns (e.g. synthetic data)
    if not cat_cols and not num_cols:
        cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in drop_cols]
        num_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in drop_cols]

    X = df.drop(columns=drop_cols)
    return X, y, cat_cols, num_cols

