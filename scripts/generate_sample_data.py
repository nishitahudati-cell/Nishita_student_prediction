"""Generate a SYNTHETIC development dataset. Not real student records."""
from pathlib import Path
import csv
import random

OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "synthetic_student_records.csv"
random.seed(42)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id", "attendance_percentage", "study_hours_per_week",
        "previous_score", "assignment_completion_rate",
        "internal_assessment_score", "previous_failures", "final_grade",
        "outcome", "data_label"
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, 151):
            attendance = round(random.uniform(45, 100), 1)
            study = round(random.uniform(1, 30), 1)
            previous = round(random.uniform(20, 95), 1)
            completion = round(random.uniform(30, 100), 1)
            internal = round(random.uniform(15, 95), 1)
            failures = random.randint(0, 3)
            # Synthetic rule creates plausible labels for software testing only.
            score = (0.22 * attendance + 0.8 * study + 0.28 * previous
                     + 0.18 * completion + 0.25 * internal - 7 * failures
                     + random.uniform(-15, 15))
            grade = max(0, min(100, round(score)))
            writer.writerow({
                "record_id": f"SYN-{i:04d}",
                "attendance_percentage": attendance,
                "study_hours_per_week": study,
                "previous_score": previous,
                "assignment_completion_rate": completion,
                "internal_assessment_score": internal,
                "previous_failures": failures,
                "final_grade": grade,
                "outcome": "Pass" if grade >= 40 else "Fail",
                "data_label": "SYNTHETIC_NOT_REAL_STUDENT_DATA",
            })
    print(f"Created clearly labelled synthetic data: {OUT}")

if __name__ == "__main__":
    main()
