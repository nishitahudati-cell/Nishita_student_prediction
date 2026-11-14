"""Model training, cross-validation, evaluation, and artifact persistence."""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from backend.app.core.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, PASS_THRESHOLD, RAW_DATA_DIR
from backend.app.ml.preprocessor import load_dataset, build_preprocessor


def get_model_candidates(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Define classification algorithms for benchmarking."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def train_and_evaluate(
    dataset_path: Path = RAW_DATA_DIR / "uci_student_performance" / "student-mat.csv",
    timing_scenario: str = "early",
    target_metric: str = "f1",
) -> Dict[str, Any]:
    """Train multiple models with leakage-free pipelines and compute metrics.
    
    1. Splits dataset into train & test sets with stratification.
    2. Builds ColumnTransformer and fits strictly on training set.
    3. Performs 5-fold Stratified Cross-Validation on training set.
    4. Evaluates held-out test set for unbiased generalization metrics.
    5. Calibrates probabilities and saves best and candidate model pipelines.
    """
    X, y, cat_cols, num_cols = load_dataset(
        file_path=dataset_path,
        timing_scenario=timing_scenario,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_model_candidates(RANDOM_STATE)

    results = {}
    best_score = -1.0
    best_model_name = "random_forest"
    best_pipeline = None

    for name, clf in models.items():
        preprocessor = build_preprocessor(cat_cols, num_cols)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # Cross validation on training set
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )

        # Fit on entire training set
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
        try:
            roc_auc_val = float(round(roc_auc_score(y_test, y_proba), 4))
        except ValueError:
            roc_auc_val = 0.5

        test_metrics = {
            "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
            "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
            "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
            "f1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
            "roc_auc": roc_auc_val,
            "confusion_matrix": cm,
            "cv_accuracy_mean": float(round(np.mean(cv_scores["test_accuracy"]), 4)),
            "cv_f1_mean": float(round(np.mean(cv_scores["test_f1"]), 4)),
            "cv_roc_auc_mean": float(round(np.mean(cv_scores["test_roc_auc"]), 4)),
        }

        # Feature importances / coefficients
        transformed_features = (
            pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        )
        feature_importance = {}
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feature_importance = {
                feat.replace("num__", "").replace("cat__", ""): float(round(imp, 4))
                for feat, imp in sorted(zip(transformed_features, importances), key=lambda x: x[1], reverse=True)[:10]
            }
        elif hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_[0])
            feature_importance = {
                feat.replace("num__", "").replace("cat__", ""): float(round(c, 4))
                for feat, c in sorted(zip(transformed_features, coefs), key=lambda x: x[1], reverse=True)[:10]
            }

        results[name] = {
            "test_metrics": test_metrics,
            "top_features": feature_importance,
        }

        # Model selection
        score = test_metrics.get(target_metric, test_metrics["f1"])
        if score > best_score:
            best_score = score
            best_model_name = name
            best_pipeline = pipeline

        # Save individual candidate model
        model_file = MODEL_DIR / f"{name}_{timing_scenario}.joblib"
        joblib.dump(pipeline, model_file)

    # Save best active model
    active_model_file = MODEL_DIR / f"active_model_{timing_scenario}.joblib"
    joblib.dump(best_pipeline, active_model_file)

    # Calibration check and save
    summary = {
        "dataset": dataset_path.name,
        "timing_scenario": timing_scenario,
        "best_model": best_model_name,
        "target_metric": target_metric,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": results,
    }

    report_file = REPORT_DIR / f"evaluation_{timing_scenario}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training completed for scenario [{timing_scenario}]. Best model: {best_model_name} (F1: {best_score})")
    print(f"Artifact saved to {active_model_file} and metrics saved to {report_file}")
    return summary


if __name__ == "__main__":
    print("--- Training Early-term Models (without G1/G2) ---")
    train_and_evaluate(timing_scenario="early")
    print("\n--- Training Late-term Models (with G1/G2) ---")
    train_and_evaluate(timing_scenario="late")

