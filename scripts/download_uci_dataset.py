"""Download and extract the public UCI Student Performance dataset."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from backend.app.core.config import RAW_DATA_DIR

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEST = RAW_DATA_DIR / "uci_student_performance"
ZIP_PATH = DEST / "student_performance.zip"

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI Student Performance dataset to: {DEST}")
    urlretrieve(URL, ZIP_PATH)
    with ZipFile(ZIP_PATH) as archive:
        archive.extractall(DEST)
    # The parent archive may contain nested zip archives (e.g., student.zip)
    for nested_zip in DEST.glob("*.zip"):
        if nested_zip != ZIP_PATH:
            with ZipFile(nested_zip) as nested_archive:
                nested_archive.extractall(DEST)
    print("Download and extraction complete.")
    print("Source: https://archive.ics.uci.edu/dataset/320/student+performance")
    print("Check the extracted CSV files before selecting a subject dataset.")

if __name__ == "__main__":
    main()
