"""Inference engine, probability estimation, risk tiering, and explainability."""
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np

from backend.app.core.config import MODEL_DIR, RAW_DATA_DIR
from backend.app.ml.dataset_schema import (
    UCI_EARLY_FEATURES,
    UCI_LATE_FEATURES,
    LEAKAGE_OR_TIMING_NOTES,
)


@lru_cache(maxsize=16)
def _load_cached_pipeline(model_path_str: str):
    """Cache loaded scikit-learn pipelines in RAM for microsecond inference."""
    return joblib.load(model_path_str)


def clear_predictor_cache() -> None:
    """Clear in-memory cached model pipelines after retraining."""
    _load_cached_pipeline.cache_clear()


class StudentPredictor:
    """Manages loaded model pipelines and produces risk-stratified predictions."""

    def __init__(self, timing_scenario: str = "early", model_name: Optional[str] = None):
        self.timing_scenario = timing_scenario
        self.model_name = model_name or "active_model"
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        filename = f"{self.model_name}_{self.timing_scenario}.joblib"
        path = MODEL_DIR / filename
        if not path.exists():
            # If active model doesn't exist yet, try random_forest or fallback
            alt_path = MODEL_DIR / f"random_forest_{self.timing_scenario}.joblib"
            if alt_path.exists():
                path = alt_path
            else:
                from backend.app.ml.trainer import train_and_evaluate
                train_and_evaluate(timing_scenario=self.timing_scenario)
                if not path.exists():
                    path = MODEL_DIR / f"active_model_{self.timing_scenario}.joblib"
        return _load_cached_pipeline(str(path.resolve()))

    def predict_single(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction, risk tier, and feature explanations for a single student."""
        df = pd.DataFrame([student_data])
        return self.predict_dataframe(df)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate predictions for a dataframe of students."""
        if df.empty:
            return []

        # Adapt generic/institutional attributes to pipeline features if present
        adapted_df = df.copy()
        if "absences" not in adapted_df.columns and "attendance_percentage" in adapted_df.columns:
            adapted_df["absences"] = adapted_df["attendance_percentage"].apply(
                lambda att: round(max(0.0, (100.0 - float(att)) / 100.0 * 60.0)) if pd.notna(att) else 4
            )
        if "studytime" not in adapted_df.columns and "study_hours_per_week" in adapted_df.columns:
            def _map_study(hrs):
                if pd.isna(hrs): return 2
                h = float(hrs)
                if h < 2: return 1
                if h < 5: return 2
                if h < 10: return 3
                return 4
            adapted_df["studytime"] = adapted_df["study_hours_per_week"].apply(_map_study)
        if "failures" not in adapted_df.columns and "previous_failures" in adapted_df.columns:
            adapted_df["failures"] = adapted_df["previous_failures"]
        if "G1" not in adapted_df.columns and "previous_score" in adapted_df.columns:
            adapted_df["G1"] = adapted_df["previous_score"].apply(lambda s: round(float(s) / 5.0, 1) if pd.notna(s) else np.nan)
        if "G2" not in adapted_df.columns and "internal_assessment_score" in adapted_df.columns:
            adapted_df["G2"] = adapted_df["internal_assessment_score"].apply(lambda s: round(float(s) / 5.0, 1) if pd.notna(s) else np.nan)

        # Inspect pipeline requirements
        pipeline_features = getattr(self.pipeline, "feature_names_in_", None)
        if pipeline_features is not None:
            features = list(pipeline_features)
        else:
            features = UCI_LATE_FEATURES if self.timing_scenario == "late" else UCI_EARLY_FEATURES
        
        # Prepare inputs with index preserved to avoid dimension collapse
        clean_df = pd.DataFrame(index=adapted_df.index)
        for col in features:
            if col in adapted_df.columns:
                clean_df[col] = adapted_df[col]
            else:
                clean_df[col] = np.nan

        if hasattr(self.pipeline, "predict_proba"):
            raw_proba = self.pipeline.predict_proba(clean_df)
            if raw_proba.ndim == 2 and raw_proba.shape[1] > 1:
                probs = raw_proba[:, 1]
            else:
                only_class = getattr(self.pipeline, "classes_", [1])[0]
                probs = np.full(len(clean_df), 1.0 if only_class == 1 else 0.0)
        else:
            probs = self.pipeline.predict(clean_df).astype(float)
        preds = self.pipeline.predict(clean_df)

        results = []
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            prob_float = float(round(prob, 4))
            outcome = "Pass" if prob_float >= 0.5 else "Fail"

            # Determine Risk Tier
            if prob_float < 0.40:
                risk_tier = "High Risk"
                recommendation = "Immediate academic advising and supplemental tutoring strongly recommended."
            elif prob_float < 0.65:
                risk_tier = "Medium Risk"
                recommendation = "Regular progress monitoring, homework support, and study group participation advised."
            else:
                risk_tier = "Low Risk"
                recommendation = "On track for successful completion. Maintain current study routine."

            # Calculate top impact factors
            row_for_factors = adapted_df.iloc[i] if i < len(adapted_df) else clean_df.iloc[i]
            factors = self._explain_prediction(row_for_factors, prob_float)

            raw_id = df.iloc[i].get("record_id", df.iloc[i].get("id", df.iloc[i].get("student_identifier", f"STU-{i+1:04d}")))
            student_id = str(raw_id) if pd.notna(raw_id) else f"STU-{i+1:04d}"

            results.append({
                "student_identifier": student_id,
                "timing_scenario": self.timing_scenario,
                "model_used": self.model_name,
                "predicted_outcome": outcome,
                "pass_probability": prob_float,
                "fail_probability": float(round(1.0 - prob_float, 4)),
                "risk_tier": risk_tier,
                "recommendation": recommendation,
                "top_factors": factors,
                "disclaimer": "Predictions are statistical estimates for academic support, not definitive judgments.",
            })

        return results

    def _explain_prediction(self, row: pd.Series, prob: float) -> List[Dict[str, Any]]:
        """Generate high-level interpretable factors contributing to the risk score."""
        factors = []

        failures = row.get("failures", row.get("previous_failures", 0))
        if pd.notna(failures) and failures > 0:
            factors.append({
                "feature": "Past Course Failures",
                "value": f"{int(failures)} failure(s)",
                "impact": "Increases Risk",
                "direction": "negative",
            })

        absences = row.get("absences", None)
        if pd.notna(absences):
            if absences > 10:
                factors.append({
                    "feature": "School Absences",
                    "value": f"{int(absences)} absences",
                    "impact": "High absenteeism elevates failure probability",
                    "direction": "negative",
                })
            elif absences <= 2:
                factors.append({
                    "feature": "School Attendance",
                    "value": f"Only {int(absences)} absences",
                    "impact": "Strong attendance correlates with higher completion",
                    "direction": "positive",
                })
        elif pd.notna(row.get("attendance_percentage")):
            att = float(row.get("attendance_percentage"))
            if att < 75.0:
                factors.append({
                    "feature": "Attendance Rate",
                    "value": f"{att}% attendance",
                    "impact": "Low attendance elevates risk of course incompletion",
                    "direction": "negative",
                })
            elif att >= 90.0:
                factors.append({
                    "feature": "Attendance Rate",
                    "value": f"{att}% attendance",
                    "impact": "High attendance commitment strongly supports success",
                    "direction": "positive",
                })

        studytime = row.get("studytime", None)
        if pd.notna(studytime):
            if studytime >= 3:
                factors.append({
                    "feature": "Weekly Study Time",
                    "value": f"Level {int(studytime)} (5+ hours/week)",
                    "impact": "Significant study time improves probability of passing",
                    "direction": "positive",
                })
            elif studytime == 1:
                factors.append({
                    "feature": "Weekly Study Time",
                    "value": "< 2 hours/week",
                    "impact": "Limited weekly study hours adds risk",
                    "direction": "negative",
                })
        elif pd.notna(row.get("study_hours_per_week")):
            sh = float(row.get("study_hours_per_week"))
            if sh >= 10:
                factors.append({
                    "feature": "Weekly Study Hours",
                    "value": f"{sh} hrs/week",
                    "impact": "Substantial weekly study commitment improves passing likelihood",
                    "direction": "positive",
                })
            elif sh < 4:
                factors.append({
                    "feature": "Weekly Study Hours",
                    "value": f"{sh} hrs/week",
                    "impact": "Low weekly study hours elevates academic risk",
                    "direction": "negative",
                })

        if self.timing_scenario == "late":
            g1 = row.get("G1")
            g2 = row.get("G2")
            if pd.notna(g1) and pd.notna(g2):
                avg_prior = (float(g1) + float(g2)) / 2.0
                if avg_prior >= 12:
                    factors.append({
                        "feature": "Prior Period Grades (G1/G2)",
                        "value": f"G1={g1}, G2={g2} (Avg: {avg_prior:.1f}/20)",
                        "impact": "Strong historical marks significantly support passing outcome",
                        "direction": "positive",
                    })
                elif avg_prior < 10:
                    factors.append({
                        "feature": "Prior Period Grades (G1/G2)",
                        "value": f"G1={g1}, G2={g2} (Avg: {avg_prior:.1f}/20)",
                        "impact": "Prior grades below threshold indicate acute academic difficulty",
                        "direction": "negative",
                    })

        higher = str(row.get("higher", "")).lower()
        if higher == "yes":
            factors.append({
                "feature": "Aspiration for Higher Education",
                "value": "Yes",
                "impact": "Desire for higher education shows positive persistence",
                "direction": "positive",
            })

        if not factors:
            factors.append({
                "feature": "Baseline Indicators",
                "value": "Standard Academic Profile",
                "impact": "Core academic indicators are within standard cohort parameters",
                "direction": "positive" if prob >= 0.5 else "negative",
            })

        return factors[:4]

