"""List and inspect CSV files in the downloaded UCI dataset folder."""
from pathlib import Path
import pandas as pd
from backend.app.core.config import RAW_DATA_DIR

DATASET_DIR = RAW_DATA_DIR / "uci_student_performance"

def main():
    if not DATASET_DIR.exists():
        raise SystemExit("Dataset not found. Run: python scripts/download_uci_dataset.py")
    csv_files = sorted(DATASET_DIR.rglob("*.csv"))
    if not csv_files:
        raise SystemExit("No CSV files found in extracted dataset.")
    for path in csv_files:
        print(f"\n--- {path.name} ---")
        frame = pd.read_csv(path, sep=";")
        print(f"Rows: {len(frame)} | Columns: {len(frame.columns)}")
        print("Columns:", ", ".join(frame.columns))
        print("Missing values:", int(frame.isna().sum().sum()))
        if "G3" in frame.columns:
            print("Pass/fail counts (threshold G3 >= 10):")
            print((frame["G3"] >= 10).map({True: "Pass", False: "Fail"}).value_counts().to_dict())

if __name__ == "__main__":
    main()
