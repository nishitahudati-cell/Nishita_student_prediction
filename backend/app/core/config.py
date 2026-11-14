from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root if it exists
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

APP_NAME = os.getenv("APP_NAME", "Student Performance Prediction System")
APP_ENV = os.getenv("APP_ENV", "development")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/student_performance.db")
PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "10"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Ensure directories exist (safe for read-only environments like Vercel)
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODEL_DIR, REPORT_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
