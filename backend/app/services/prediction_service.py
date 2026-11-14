"""Prediction service orchestrating inference and audit persistence."""
import io
import pandas as pd
from typing import Dict, Any, List, Optional

from backend.app.ml.predictor import StudentPredictor
from backend.app.database.repository import save_prediction, save_predictions_bulk
from backend.app.schemas.student import StudentInput


def predict_single_student(
    student: StudentInput,
    timing_scenario: str = "early",
    model_name: Optional[str] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Execute prediction pipeline for one student and record in database."""
    predictor = StudentPredictor(timing_scenario=timing_scenario, model_name=model_name)
    input_dict = student.model_dump(exclude_none=True)
    
    result = predictor.predict_single(input_dict)
    
    if persist:
        save_prediction(
            student_identifier=result["student_identifier"],
            timing_scenario=timing_scenario,
            model_used=result["model_used"],
            inputs=input_dict,
            predicted_outcome=result["predicted_outcome"],
            pass_probability=result["pass_probability"],
            risk_tier=result["risk_tier"],
            top_factors=result["top_factors"],
        )
        try:
            from backend.app.api.websocket import manager
            import asyncio
            if manager.active_connections:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(manager.broadcast({
                        "type": "new_prediction",
                        "student_identifier": result["student_identifier"],
                        "predicted_outcome": result["predicted_outcome"],
                        "pass_probability": result["pass_probability"],
                        "risk_tier": result["risk_tier"],
                    }))
        except Exception:
            pass
    return result


def predict_batch_csv(
    file_bytes: bytes,
    timing_scenario: str = "early",
    model_name: Optional[str] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Execute batch predictions from uploaded CSV content."""
    # Detect separator across semicolons, commas, and tabs
    sample_text = file_bytes[:2048].decode("utf-8", errors="ignore")
    counts = {
        ";": sample_text.count(";"),
        ",": sample_text.count(","),
        "\t": sample_text.count("\t"),
    }
    sep = max(counts, key=counts.get) if any(counts.values()) else ","
    
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding="latin-1")
        except Exception:
            df = pd.DataFrame()
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        return {
            "total_records": 0,
            "pass_count": 0,
            "fail_count": 0,
            "risk_breakdown": {
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
            },
            "predictions": [],
        }

    predictor = StudentPredictor(timing_scenario=timing_scenario, model_name=model_name)
    records = predictor.predict_dataframe(df)
    
    if persist:
        bulk_records = []
        for i, r in enumerate(records):
            row_dict = df.iloc[i].to_dict() if i < len(df) else {}
            clean_inputs = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
            bulk_records.append({
                "student_identifier": r["student_identifier"],
                "timing_scenario": timing_scenario,
                "model_used": r["model_used"],
                "inputs": clean_inputs,
                "predicted_outcome": r["predicted_outcome"],
                "pass_probability": r["pass_probability"],
                "risk_tier": r["risk_tier"],
                "top_factors": r["top_factors"],
            })
        save_predictions_bulk(bulk_records)

    # Compute summary aggregates
    total = len(records)
    pass_count = sum(1 for r in records if r["predicted_outcome"] == "Pass")
    fail_count = total - pass_count
    high_risk = sum(1 for r in records if r["risk_tier"] == "High Risk")
    medium_risk = sum(1 for r in records if r["risk_tier"] == "Medium Risk")
    low_risk = sum(1 for r in records if r["risk_tier"] == "Low Risk")

    return {
        "total_records": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "risk_breakdown": {
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
        },
        "predictions": records,
    }


def simulate_intervention(
    student: StudentInput,
    interventions: Dict[str, Any],
    timing_scenario: str = "early",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare baseline student prediction against simulated intervention scenario."""
    # 1. Baseline prediction (without persisting to audit table)
    base_res = predict_single_student(
        student=student,
        timing_scenario=timing_scenario,
        model_name=model_name,
        persist=False,
    )

    # 2. Apply intervention parameter updates
    sim_data = student.model_dump(exclude_none=True)
    for k, v in interventions.items():
        if v is not None:
            sim_data[k] = v

    sim_student = StudentInput(**sim_data)
    sim_res = predict_single_student(
        student=sim_student,
        timing_scenario=timing_scenario,
        model_name=model_name,
        persist=False,
    )

    prob_delta = round(float(sim_res["pass_probability"] - base_res["pass_probability"]), 4)
    percentage_gain = round(prob_delta * 100.0, 1)

    return {
        "baseline": base_res,
        "simulated": sim_res,
        "probability_delta": prob_delta,
        "percentage_gain": percentage_gain,
        "risk_migration": f"{base_res['risk_tier']} -> {sim_res['risk_tier']}",
        "improved": prob_delta > 0,
    }

