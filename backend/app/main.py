"""FastAPI application entry point."""
import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.app.core.config import APP_NAME, MODEL_DIR, PROJECT_ROOT
from backend.app.database.session import init_db
from backend.app.api.health import router as health_router
from backend.app.api.predict import router as predict_router
from backend.app.api.train import router as train_router
from backend.app.api.history import router as history_router
from backend.app.api.students import router as students_router
from backend.app.api.analytics import router as analytics_router
from backend.app.api.auth import router as auth_router
from backend.app.api.websocket import router as ws_router

logger = logging.getLogger("apex_analytics")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    # Initialize SQLite database with WAL mode and performance indexes
    init_db()

    # Verify models exist; train if missing
    early_model = MODEL_DIR / "active_model_early.joblib"
    if not early_model.exists():
        from backend.app.ml.trainer import train_and_evaluate
        train_and_evaluate(timing_scenario="early")
        train_and_evaluate(timing_scenario="late")

    yield


tags_metadata = [
    {"name": "Health", "description": "Core service readiness, database connectivity, and environment diagnostics."},
    {"name": "Authentication & Access Control", "description": "Institutional advisor login, role credentials, and session token verification."},
    {"name": "Student Management", "description": "Authenticated UCI benchmark student registry, pagination, search, and comparative analysis."},
    {"name": "Prediction", "description": "Single-instance inference, batch CSV scoring, and What-If intervention simulation."},
    {"name": "Analytics & Cohort Reports", "description": "Cohort performance metrics, demographic impact, and longitudinal distributions."},
    {"name": "History & Audit", "description": "Logged inference audit records and statistical aggregates."},
    {"name": "Training & Benchmarks", "description": "Model training orchestration, cross-validation metrics, and serialized artifacts."},
    {"name": "Real-Time WebSockets", "description": "Bi-directional WebSocket streaming for live inference events and dashboard feeds."},
]

app = FastAPI(
    title=APP_NAME,
    description="Modular academic analytics and machine learning system for student performance prediction.",
    version="1.2.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    """Inject request ID and execution latency header into all HTTP responses."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format schema validation errors into structured, client-friendly responses."""
    safe_errors = jsonable_encoder(exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "status_code": 422,
            "details": safe_errors,
            "detail": safe_errors,
            "message": "The request payload failed validation against expected schema.",
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Gracefully handle missing files or datasets."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource Not Found",
            "status_code": 404,
            "message": str(exc),
        },
    )


# Include API Routers
app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(train_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(students_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(ws_router, prefix="/api")

# Static / Dashboard UI mount
STATIC_DIR = PROJECT_ROOT / "backend" / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
@app.head("/", include_in_schema=False)
def index():
    """Serve the interactive Education Analytics dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Student Performance Prediction System API is running. Visit <a href='/docs'>/docs</a>.</h1>")
