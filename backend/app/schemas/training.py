"""Pydantic schemas for training triggers and evaluation reports."""
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    timing_scenario: Literal["early", "late"] = Field(default="early", description="Feature scenario: early (pre-grades) or late (with G1/G2)")
    dataset: Literal["student-mat.csv", "student-por.csv", "synthetic_student_records.csv"] = Field(
        default="student-mat.csv",
        description="Dataset to train upon"
    )
    target_metric: Literal["f1", "accuracy", "roc_auc"] = Field(default="f1", description="Metric to optimize for model selection")


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    cv_accuracy_mean: float
    cv_f1_mean: float
    cv_roc_auc_mean: float


class ModelEvaluationSummary(BaseModel):
    dataset: str
    timing_scenario: str
    best_model: str
    target_metric: str
    train_samples: int
    test_samples: int
    models: Dict[str, Any]

