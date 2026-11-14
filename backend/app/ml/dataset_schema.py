
"""Dataset metadata, feature sets, and leakage-aware target definitions."""

TARGET_COLUMN = "G3"
PASS_THRESHOLD = 10
TARGET_DESCRIPTION = "Pass when final grade G3 >= 10 (UCI grade scale 0-20)."

UCI_CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

UCI_NUMERIC_FEATURES_EARLY = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]

UCI_NUMERIC_FEATURES_LATE = UCI_NUMERIC_FEATURES_EARLY + ["G1", "G2"]

UCI_EARLY_FEATURES = UCI_CATEGORICAL_FEATURES + UCI_NUMERIC_FEATURES_EARLY
UCI_LATE_FEATURES = UCI_CATEGORICAL_FEATURES + UCI_NUMERIC_FEATURES_LATE

UCI_MATH_EXPECTED_COLUMNS = UCI_LATE_FEATURES + [TARGET_COLUMN]

LEAKAGE_OR_TIMING_NOTES = {
    "G3": "Target only; never include in predictors.",
    "G1": "First-period grade (0-20); eligible only in late-term scenario.",
    "G2": "Second-period grade (0-20); eligible only in late-term scenario.",
    "absences": "Number of school absences (0-93); count indicator.",
}

# Generic / Institutional feature contract (for synthetic or non-UCI institutional records)
GENERIC_NUMERIC_FEATURES = [
    "attendance_percentage",
    "study_hours_per_week",
    "previous_score",
    "assignment_completion_rate",
    "internal_assessment_score",
    "previous_failures",
]
