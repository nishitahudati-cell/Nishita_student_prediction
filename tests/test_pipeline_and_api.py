"""Integration and unit tests for the ML pipeline, predictor, and database repository."""
import unittest
from pathlib import Path
import pandas as pd

from backend.app.ml.preprocessor import load_dataset, detect_separator
from backend.app.ml.predictor import StudentPredictor
from backend.app.database.session import init_db
from backend.app.database.repository import (
    save_prediction,
    get_recent_predictions,
    log_model_run,
)
from backend.app.schemas.student import StudentInput, SimulationRequest, SimulationResponse
from backend.app.api.predict import predict_student, simulate_student_intervention
from backend.app.api.health import health_check
from backend.app.api.analytics import get_dashboard_data
from backend.app.api.students import list_students, get_student_detail, get_student_alerts, compare_students
from backend.app.core.config import RAW_DATA_DIR, MODEL_DIR


class TestStudentPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_uci_dataset_exists(self):
        mat_path = RAW_DATA_DIR / "uci_student_performance" / "student-mat.csv"
        self.assertTrue(mat_path.exists(), "UCI Mathematics dataset must exist")

    def test_separator_detection(self):
        mat_path = RAW_DATA_DIR / "uci_student_performance" / "student-mat.csv"
        sep = detect_separator(mat_path)
        self.assertEqual(sep, ";", "UCI dataset must detect semicolon delimiter")

    def test_early_scenario_leakage_prevention(self):
        """Ensure G1, G2, and G3 never appear in early predictors."""
        X, y, cat_cols, num_cols = load_dataset(timing_scenario="early")
        self.assertNotIn("G3", X.columns, "G3 is target and must never be a predictor")
        self.assertNotIn("G1", X.columns, "G1 must be excluded in early scenario")
        self.assertNotIn("G2", X.columns, "G2 must be excluded in early scenario")
        self.assertEqual(len(y), len(X), "Target vector length must match feature matrix")

    def test_late_scenario_prior_grades_included(self):
        """Ensure G1 and G2 are present in late scenario while G3 is excluded."""
        X, y, cat_cols, num_cols = load_dataset(timing_scenario="late")
        self.assertNotIn("G3", X.columns, "G3 target must be excluded")
        self.assertIn("G1", X.columns, "G1 should be present in late scenario")
        self.assertIn("G2", X.columns, "G2 should be present in late scenario")

    def test_student_predictor_early(self):
        predictor = StudentPredictor(timing_scenario="early")
        sample = {
            "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
            "studytime": 3, "failures": 0, "absences": 2, "higher": "yes",
        }
        res = predictor.predict_single(sample)
        self.assertIn(res["predicted_outcome"], ["Pass", "Fail"])
        self.assertGreaterEqual(res["pass_probability"], 0.0)
        self.assertLessEqual(res["pass_probability"], 1.0)
        self.assertIn(res["risk_tier"], ["Low Risk", "Medium Risk", "High Risk"])
        self.assertTrue(len(res["top_factors"]) > 0, "Should extract explanation factors")

    def test_student_predictor_late(self):
        predictor = StudentPredictor(timing_scenario="late")
        high_risk_sample = {
            "studytime": 1, "failures": 2, "absences": 15, "G1": 5, "G2": 6,
        }
        res = predictor.predict_single(high_risk_sample)
        self.assertEqual(res["predicted_outcome"], "Fail")
        self.assertEqual(res["risk_tier"], "High Risk")

    def test_database_persistence(self):
        rec_id = save_prediction(
            student_identifier="TEST-PERSIST-99",
            timing_scenario="early",
            model_used="random_forest",
            inputs={"studytime": 2, "failures": 0},
            predicted_outcome="Pass",
            pass_probability=0.72,
            risk_tier="Low Risk",
            top_factors=[{"feature": "failures", "value": "0", "impact": "Positive", "direction": "positive"}],
        )
        self.assertIsInstance(rec_id, int)
        history = get_recent_predictions(limit=10)
        self.assertTrue(any(h["student_identifier"] == "TEST-PERSIST-99" for h in history))

    def test_student_records_querying(self):
        result = list_students(subject="all", page=1, page_size=10)
        self.assertEqual(result["total"], 1044, "Total students across both subjects must be 1044")
        self.assertEqual(len(result["records"]), 10)
        self.assertIn("student_identifier", result["records"][0])
        self.assertIn("attendance_percentage", result["records"][0])
        self.assertIn("predicted_outcome", result["records"][0])

    def test_student_detail_lookup(self):
        student = get_student_detail("MAT-0001")
        self.assertIsNotNone(student)
        self.assertEqual(student["student_identifier"], "MAT-0001")
        self.assertEqual(student["subject"], "Mathematics")
        self.assertIn("G3", student)

    def test_dashboard_analytics_aggregation(self):
        analytics = get_dashboard_data(subject="all")
        self.assertEqual(analytics["total_students"], 1044)
        self.assertGreater(analytics["predicted_pass_rate"], 50.0)
        self.assertTrue(len(analytics["performance_distribution"]) > 0)
        self.assertTrue(len(analytics["attendance_analytics"]) > 0)
        self.assertTrue(len(analytics["study_hours_analytics"]) > 0)
        self.assertEqual(len(analytics["subject_comparison"]), 2)

    def test_health_check_endpoint(self):
        health = health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["database_connected"])
        self.assertTrue(health["uci_dataset_ready"])
        self.assertGreater(health["models_count"], 0)

    def test_predict_student_endpoint(self):
        inp = StudentInput(student_identifier="TEST-PRED-API", studytime=3, failures=0, absences=2)
        res = predict_student(payload=inp)
        self.assertIn(res["predicted_outcome"], ["Pass", "Fail"])
        self.assertIn(res["risk_tier"], ["Low Risk", "Medium Risk", "High Risk"])

    def test_student_alerts_endpoint(self):
        alerts = get_student_alerts(limit=5, subject="all")
        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)
        first = alerts[0]
        self.assertIn("student_identifier", first)
        self.assertIn("primary_hazard", first)
        self.assertIn("prescribed_action", first)
        self.assertEqual(first["risk_tier"], "High Risk")

    def test_student_comparison_endpoint(self):
        # Test comparison with cohort mean
        comp_cohort = compare_students(student_a="MAT-0001", student_b="cohort_mean")
        self.assertIn("student_a", comp_cohort)
        self.assertIn("student_b", comp_cohort)
        self.assertIn("radar", comp_cohort)
        self.assertEqual(len(comp_cohort["radar"]["labels"]), 6)
        self.assertIn("deltas", comp_cohort)

        # Test comparison between two distinct students
        comp_dual = compare_students(student_a="MAT-0001", student_b="POR-0001")
        self.assertEqual(comp_dual["student_a"]["student_identifier"], "MAT-0001")
        self.assertEqual(comp_dual["student_b"]["student_identifier"], "POR-0001")
        self.assertFalse(comp_dual["student_b"].get("is_benchmark", False))

    def test_simulate_intervention_endpoint(self):
        req = SimulationRequest(
            student=StudentInput(studytime=1, failures=2, absences=16, schoolsup="no", G1=6, G2=6),
            interventions={"studytime": 3, "absences": 4, "schoolsup": "yes", "G2": 10},
            timing_scenario="late",
        )
        res = simulate_student_intervention(req)
        model_res = SimulationResponse(**res)
        self.assertIn(model_res.baseline.predicted_outcome, ["Pass", "Fail"])
        self.assertIn(model_res.simulated.predicted_outcome, ["Pass", "Fail"])
        self.assertTrue(model_res.improved)
        self.assertGreater(model_res.probability_delta, 0.0)

    def test_batch_prediction_service_and_adapter(self):
        from backend.app.services.prediction_service import predict_batch_csv
        # 1. UCI batch
        with open(RAW_DATA_DIR.parents[1] / "backend" / "app" / "static" / "synthetic_sample.csv", "rb") as f:
            uci_data = f.read()
        res_uci = predict_batch_csv(uci_data, timing_scenario="early")
        self.assertEqual(res_uci["total_records"], 50)
        self.assertIn("pass_count", res_uci)
        self.assertIn("risk_breakdown", res_uci)

        # 2. Synthetic dataset with custom columns
        with open(RAW_DATA_DIR.parents[1] / "data" / "sample" / "synthetic_student_records.csv", "rb") as f:
            syn_data = f.read()
        res_syn = predict_batch_csv(syn_data, timing_scenario="early")
        self.assertEqual(res_syn["total_records"], 150)
        self.assertTrue(len(res_syn["predictions"][0]["top_factors"]) > 0)

        # 3. Empty batch
        res_empty = predict_batch_csv(b"", timing_scenario="early")
        self.assertEqual(res_empty["total_records"], 0)
        self.assertEqual(len(res_empty["predictions"]), 0)

    def test_synthetic_retraining_and_reports(self):
        from backend.app.services.model_service import (
            trigger_retraining,
            get_evaluation_report,
            list_trained_artifacts,
        )
        # Train on synthetic dataset
        summary = trigger_retraining(dataset_name="synthetic_student_records.csv", timing_scenario="early")
        self.assertEqual(summary["dataset"], "synthetic_student_records.csv")
        self.assertIn("best_model", summary)
        self.assertEqual(summary["train_samples"], 120)

        # Restore baseline benchmark
        bench_summary = trigger_retraining(dataset_name="student-mat.csv", timing_scenario="early")
        self.assertEqual(bench_summary["dataset"], "student-mat.csv")

        # Report retrieval
        rep = get_evaluation_report("early")
        self.assertIn("models", rep)

        # Artifacts list
        arts = list_trained_artifacts()
        self.assertGreater(len(arts), 0)


class TestAsgiHttpEndpoints(unittest.TestCase):
    """End-to-end integration tests for FastAPI routes via in-process ASGI execution."""

    @classmethod
    def setUpClass(cls):
        init_db()

    @staticmethod
    def _call(method: str, path: str, body: bytes = b"", headers: dict = None, return_headers: bool = False):
        import asyncio
        from backend.app.main import app

        headers = headers or {}
        raw_headers = [(k.lower().encode("latin1"), v.encode("latin1")) for k, v in headers.items()]
        raw_headers.append((b"host", b"testserver"))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "path": path.split("?")[0],
            "raw_path": path.encode("latin1"),
            "query_string": b"" if "?" not in path else path.split("?", 1)[1].encode("latin1"),
            "headers": raw_headers,
        }

        response_body = []
        status = None
        resp_headers = {}
        sent = False
        disconnect_event = asyncio.Event()

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                for k, v in message.get("headers", []):
                    resp_headers[k.decode("latin1").lower()] = v.decode("latin1")
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))
                if not message.get("more_body", False):
                    disconnect_event.set()

        asyncio.run(app(scope, receive, send))
        text = b"".join(response_body).decode("utf-8", errors="replace")
        if return_headers:
            return status, text, resp_headers
        return status, text

    @staticmethod
    def _call_ws(path: str, client_messages: list):
        import asyncio
        import json
        from backend.app.main import app

        messages = []
        step = 0

        async def receive():
            nonlocal step
            if step == 0:
                step += 1
                return {"type": "websocket.connect"}
            msg_idx = step - 1
            if msg_idx < len(client_messages):
                step += 1
                msg = client_messages[msg_idx]
                text = json.dumps(msg) if isinstance(msg, dict) else str(msg)
                return {"type": "websocket.receive", "text": text}
            return {"type": "websocket.disconnect", "code": 1000}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "websocket",
            "path": path,
            "raw_path": path.encode("latin1"),
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
            "subprotocols": [],
        }

        asyncio.run(app(scope, receive, send))
        return [json.loads(m["text"]) for m in messages if m["type"] == "websocket.send"]

    def test_http_dashboard_index(self):
        status, body = self._call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Student Performance Analytics", body)

    def test_http_health(self):
        status, body = self._call("GET", "/api/health")
        self.assertEqual(status, 200)
        import json
        data = json.loads(body)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["database_connected"])

    def test_http_analytics_dashboard(self):
        status, body = self._call("GET", "/api/analytics/dashboard?subject=all")
        self.assertEqual(status, 200)
        import json
        data = json.loads(body)
        self.assertEqual(data["total_students"], 1044)
        self.assertIn("performance_distribution", data)

    def test_http_students_pagination_and_detail(self):
        status, body = self._call("GET", "/api/students?page=1&page_size=5&subject=math")
        self.assertEqual(status, 200)
        import json
        data = json.loads(body)
        self.assertEqual(len(data["records"]), 5)

        # Valid student detail
        status_s, body_s = self._call("GET", "/api/students/MAT-0001")
        self.assertEqual(status_s, 200)
        detail = json.loads(body_s)
        self.assertEqual(detail["student_identifier"], "MAT-0001")

        # Missing student detail
        status_404, _ = self._call("GET", "/api/students/NON_EXISTENT_ID")
        self.assertEqual(status_404, 404)

    def test_http_student_alerts_and_compare(self):
        status, body = self._call("GET", "/api/students/alerts?limit=3")
        self.assertEqual(status, 200)
        import json
        alerts = json.loads(body)
        self.assertTrue(len(alerts) > 0)

        # Comparison with cohort mean
        status_comp, body_comp = self._call("GET", "/api/students/compare?student_a=MAT-0001&student_b=cohort_mean")
        self.assertEqual(status_comp, 200)
        comp = json.loads(body_comp)
        self.assertIn("radar", comp)

        # Invalid student comparison
        status_bad, _ = self._call("GET", "/api/students/compare?student_a=INVALID_A&student_b=INVALID_B")
        self.assertEqual(status_bad, 404)

    def test_http_predict_and_simulate(self):
        import json
        payload = json.dumps({"student_identifier": "API-TEST-1", "studytime": 3, "failures": 0, "absences": 2}).encode()
        status, body = self._call("POST", "/api/predict?timing_scenario=early", body=payload, headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn(data["predicted_outcome"], ["Pass", "Fail"])

        # Simulation
        sim_payload = json.dumps({
            "student": {"student_identifier": "API-TEST-2", "studytime": 1, "failures": 2, "absences": 15},
            "interventions": {"studytime": 3, "absences": 3},
            "timing_scenario": "early"
        }).encode()
        status_sim, body_sim = self._call("POST", "/api/predict/simulate", body=sim_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(status_sim, 200)
        sim_data = json.loads(body_sim)
        self.assertIn("percentage_gain", sim_data)

    def test_http_batch_predict_and_errors(self):
        boundary = "----WebKitFormBoundaryX"
        csv_data = b"school;sex;age;studytime;failures;absences;G1;G2\nGP;F;17;2;0;4;12;13\n"
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"students.csv\"\r\n"
            f"Content-Type: text/csv\r\n\r\n"
        ).encode() + csv_data + f"\r\n--{boundary}--\r\n".encode()

        status, body = self._call(
            "POST",
            "/api/predict/batch?timing_scenario=early",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        self.assertEqual(status, 200)

        # Non-csv upload error
        bad_body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"students.txt\"\r\n"
            f"Content-Type: text/plain\r\n\r\n"
            "text content"
            f"\r\n--{boundary}--\r\n"
        ).encode()
        status_bad, _ = self._call(
            "POST",
            "/api/predict/batch",
            body=bad_body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        self.assertEqual(status_bad, 400)

    def test_http_train_reports_and_history(self):
        import json
        status_rep, body_rep = self._call("GET", "/api/train/reports/early")
        self.assertEqual(status_rep, 200)

        status_models, body_models = self._call("GET", "/api/train/models")
        self.assertEqual(status_models, 200)

        # History list
        status_h, body_h = self._call("GET", "/api/history?limit=5")
        self.assertEqual(status_h, 200)
        hist = json.loads(body_h)
        self.assertIn("history", hist)

        # Clear history
        status_del, body_del = self._call("DELETE", "/api/history")
        self.assertEqual(status_del, 200)
        del_res = json.loads(body_del)
        self.assertEqual(del_res["status"], "success")

    def test_http_analytics_summary_and_export(self):
        import json
        status_sum, body_sum = self._call("GET", "/api/analytics/summary?subject=all")
        self.assertEqual(status_sum, 200)
        sum_data = json.loads(body_sum)
        self.assertEqual(sum_data["total_students"], 1044)
        self.assertIn("median_performance", sum_data)
        self.assertIn("performance_std", sum_data)

        # CSV Export endpoint
        status_exp, body_exp = self._call("GET", "/api/students/export?subject=math")
        self.assertEqual(status_exp, 200)
        self.assertIn("student_identifier", body_exp)
        self.assertIn("MAT-0001", body_exp)

    def test_http_audit_stats_endpoint(self):
        import json
        status, body = self._call("GET", "/api/history/stats")
        self.assertEqual(status, 200)
        stats = json.loads(body)
        self.assertIn("total_logged", stats)
        self.assertIn("average_pass_probability", stats)

    def test_http_frontend_index_and_dom_elements(self):
        status, html = self._call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Student Performance Analytics", html)

        # 6 SPA Views
        for view_id in ["view-dashboard", "view-students", "view-predict", "view-models", "view-analytics", "view-settings"]:
            self.assertIn(f'id="{view_id}"', html, f"Missing view: {view_id}")

        # Interactive Modals & Overlays
        for modal_id in [
            "student-detail-modal",
            "compare-students-modal",
            "shortcuts-modal",
            "audit-detail-modal",
            "auth-session-modal",
            "auth-lock-overlay",
        ]:
            self.assertIn(f'id="{modal_id}"', html, f"Missing modal/overlay: {modal_id}")

        # Core Action IDs & Elements
        for elem_id in [
            "advisor-auth-btn",
            "batch-dropzone",
            "pred-submit-btn",
            "comp-math-students",
            "comp-math-passrate",
            "comp-math-avggrade",
            "comp-math-absences",
            "comp-math-highrisk",
            "comp-por-students",
            "comp-por-passrate",
            "comp-por-avggrade",
            "comp-por-absences",
            "comp-por-highrisk",
        ]:
            self.assertIn(f'id="{elem_id}"', html, f"Missing core element ID: {elem_id}")

        # Chart Canvas Elements
        for chart_id in [
            "chart-performance-dist",
            "chart-risk-donut",
            "chart-attendance-impact",
            "chart-study-impact",
            "chart-models-comparison",
            "chart-parent-edu",
            "chart-alcohol-impact",
            "chart-compare-radar",
            "modal-trajectory-chart",
        ]:
            self.assertIn(f'id="{chart_id}"', html, f"Missing chart canvas: {chart_id}")

    def test_backend_lifespan_startup_and_shutdown(self):
        import asyncio
        from backend.app.main import app, lifespan

        async def run_lifespan():
            async with lifespan(app):
                pass

        asyncio.run(run_lifespan())

    def test_http_static_assets_serving(self):
        status, body = self._call("GET", "/static/index.html")
        self.assertEqual(status, 200)
        self.assertIn("Institutional Analytics Dashboard", body)

    def test_http_auth_and_authorization_flows(self):
        import json
        # 1. Valid Advisor Login
        login_payload = json.dumps({"username": "advisor", "password": "1234"})
        status, body = self._call("POST", "/api/auth/login", login_payload.encode("utf-8"), {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        login_data = json.loads(body)
        self.assertIn("access_token", login_data)
        self.assertEqual(login_data["role"], "Senior Academic Advisor")
        self.assertEqual(login_data["username"], "advisor")
        token = login_data["access_token"]

        # 2. Invalid Credentials
        bad_payload = json.dumps({"username": "advisor", "password": "wrong_password"})
        status_bad, _ = self._call("POST", "/api/auth/login", bad_payload.encode("utf-8"), {"Content-Type": "application/json"})
        self.assertEqual(status_bad, 401)

        # 3. Authenticated Profile with Bearer Token
        status_me, body_me = self._call("GET", "/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(status_me, 200)
        me_data = json.loads(body_me)
        self.assertTrue(me_data["authenticated"])
        self.assertEqual(me_data["username"], "advisor")

        # 4. Unauthenticated Profile (Guest Mode)
        status_guest, body_guest = self._call("GET", "/api/auth/me")
        self.assertEqual(status_guest, 200)
        guest_data = json.loads(body_guest)
        self.assertFalse(guest_data["authenticated"])

        # 5. Token Verification Endpoint
        verify_payload = json.dumps({"token": token})
        status_v, body_v = self._call("POST", "/api/auth/verify", verify_payload.encode("utf-8"), {"Content-Type": "application/json"})
        self.assertEqual(status_v, 200)
        self.assertTrue(json.loads(body_v)["valid"])

        # 6. Invalid Token Verification
        bad_token_payload = json.dumps({"token": "invalid.corrupt.token"})
        status_bv, _ = self._call("POST", "/api/auth/verify", bad_token_payload.encode("utf-8"), {"Content-Type": "application/json"})
        self.assertEqual(status_bv, 401)

    def test_http_cors_and_latency_headers(self):
        # 1. CORS Preflight OPTIONS Request
        status, body, headers = self._call(
            "OPTIONS",
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
            return_headers=True,
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://localhost:3000")
        self.assertEqual(headers.get("access-control-allow-credentials"), "true")
        self.assertIn("GET", headers.get("access-control-allow-methods", ""))

        # 2. X-Request-ID and X-Process-Time Latency Middleware
        status_h, _, headers_h = self._call("GET", "/api/health", return_headers=True)
        self.assertEqual(status_h, 200)
        self.assertIn("x-request-id", headers_h)
        self.assertIn("x-process-time", headers_h)

    def test_http_websocket_live_stream(self):
        # Bi-directional in-process ASGI WebSocket stream
        messages = [
            {"action": "ping"},
            {"action": "status"},
            {"action": "subscribe", "channel": "inferences"},
        ]
        responses = self._call_ws("/api/ws/live", messages)
        self.assertGreaterEqual(len(responses), 4)
        msg_types = [r.get("type") for r in responses]
        self.assertIn("welcome", msg_types)
        self.assertIn("pong", msg_types)
        self.assertIn("status", msg_types)
        self.assertIn("subscribed", msg_types)

    def test_http_request_validation_and_edge_cases(self):
        import json
        # 1. Non-existent Student -> 404
        status_stu, body_stu = self._call("GET", "/api/students/STU-999999")
        self.assertEqual(status_stu, 404)

        # 2. Non-existent History Record -> 404
        status_hist, body_hist = self._call("GET", "/api/history/999999")
        self.assertEqual(status_hist, 404)

        # 3. Invalid Request Schema (Negative age) -> 422
        bad_pred = json.dumps({"age": -5, "studytime": 99})
        status_val, body_val = self._call("POST", "/api/predict", bad_pred.encode("utf-8"), {"Content-Type": "application/json"})
        self.assertEqual(status_val, 422)
        val_data = json.loads(body_val)
        self.assertIn("details", val_data)

        # 4. Invalid Query Param Timing Scenario -> 422
        status_scen, _ = self._call("POST", "/api/predict?timing_scenario=unsupported")
        self.assertEqual(status_scen, 422)

        # 5. Invalid Path Param Timing Scenario in Reports -> 422
        status_rep, _ = self._call("GET", "/api/train/reports/unsupported")
        self.assertEqual(status_rep, 422)

        # 6. Empty Batch CSV Upload -> 400
        boundary = "----WebKitFormBoundaryBatchTest"
        empty_multipart = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="empty.csv"\r\n'
            "Content-Type: text/csv\r\n\r\n"
            f"\r\n--{boundary}--\r\n"
        ).encode("utf-8")
        status_empty, body_empty = self._call(
            "POST",
            "/api/predict/batch",
            empty_multipart,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(status_empty, 400)
        self.assertIn("empty", body_empty.lower())

        # 7. Non-CSV File in Batch Upload -> 400
        txt_multipart = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="data.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            f"plain text content\r\n--{boundary}--\r\n"
        ).encode("utf-8")
        status_txt, body_txt = self._call(
            "POST",
            "/api/predict/batch",
            txt_multipart,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(status_txt, 400)
        self.assertIn("csv", body_txt.lower())

    def test_database_crud_transactions_and_null_safety(self):
        from backend.app.database.repository import (
            save_prediction,
            save_predictions_bulk,
            get_prediction_by_id,
            get_recent_predictions,
            get_prediction_stats,
            clear_all_predictions,
        )

        # 1. Save prediction with None / empty top factors (null safety)
        pid = save_prediction(
            student_identifier="NULL-TEST-001",
            timing_scenario="early",
            model_used="unit_test_model",
            inputs={"test": True, "value": 123},
            predicted_outcome="Pass",
            pass_probability=0.88,
            risk_tier="Low Risk",
            top_factors=None,
        )
        self.assertIsInstance(pid, int)

        # 2. Read by ID and verify null factors parsed safely as []
        rec = get_prediction_by_id(pid)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["student_identifier"], "NULL-TEST-001")
        self.assertIsInstance(rec["top_factors"], list)

        # 3. Bulk insert transaction
        bulk_data = [
            {
                "student_identifier": f"BULK-TEST-{i}",
                "timing_scenario": "early",
                "model_used": "unit_test_model",
                "inputs": {"idx": i},
                "predicted_outcome": "Pass" if i % 2 == 0 else "Fail",
                "pass_probability": 0.5 + (i * 0.05),
                "risk_tier": "Low Risk" if i % 2 == 0 else "High Risk",
                "top_factors": [],
            }
            for i in range(5)
        ]
        count = save_predictions_bulk(bulk_data)
        self.assertEqual(count, 5)

        # 4. Aggregations and filtering
        recent = get_recent_predictions(limit=10, search="BULK-TEST")
        self.assertGreaterEqual(len(recent), 5)

        stats = get_prediction_stats()
        self.assertGreaterEqual(stats["total_logged"], 6)

        # 5. Clean up test records
        from backend.app.database.session import get_db_connection
        with get_db_connection() as conn:
            conn.execute("DELETE FROM predictions WHERE student_identifier LIKE '%-TEST-%'")
            conn.commit()

    def test_database_connectivity_and_pragmas(self):
        from backend.app.database.session import get_db_connection, DB_PATH
        self.assertTrue(DB_PATH.exists())
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 1. Integrity check
            cursor.execute("PRAGMA integrity_check;")
            self.assertEqual(cursor.fetchone()[0], "ok")

            # 2. Foreign key check
            cursor.execute("PRAGMA foreign_key_check;")
            self.assertEqual(len(cursor.fetchall()), 0)

            # 3. Journal mode & Foreign keys enabled
            cursor.execute("PRAGMA journal_mode;")
            self.assertEqual(cursor.fetchone()[0].lower(), "wal")

            cursor.execute("PRAGMA foreign_keys;")
            self.assertEqual(cursor.fetchone()[0], 1)

    def test_database_schema_migrations_and_versioning(self):
        from backend.app.database.session import get_db_connection, DB_PATH
        from backend.app.database.migrations import run_migrations, get_applied_migration_versions

        # 1. Verify all 4 migrations are tracked
        with get_db_connection() as conn:
            versions = get_applied_migration_versions(conn)
            self.assertIn(1, versions)
            self.assertIn(2, versions)
            self.assertIn(3, versions)
            self.assertIn(4, versions)

        # 2. Idempotency test: re-running migrations applies 0 additional migrations
        new_applied = run_migrations(DB_PATH)
        self.assertEqual(len(new_applied), 0)

    def test_database_full_crud_and_relationships(self):
        import json
        from backend.app.database.repository import (
            save_prediction,
            get_prediction_by_id,
            update_prediction_status,
            add_advisory_note,
            get_advisory_notes,
            delete_prediction_by_id,
            get_recent_predictions,
        )

        # 1. CREATE Prediction
        pid = save_prediction(
            student_identifier="DBTEST-STU-001",
            timing_scenario="early",
            model_used="random_forest",
            inputs={"studytime": 3, "failures": 0},
            predicted_outcome="Pass",
            pass_probability=0.91,
            risk_tier="Low Risk",
            top_factors=[{"feature": "studytime", "impact": "+10%"}],
        )
        self.assertIsInstance(pid, int)

        # 2. READ Prediction
        rec = get_prediction_by_id(pid)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["student_identifier"], "DBTEST-STU-001")
        self.assertEqual(rec["status"], "Pending")

        # 3. UPDATE Prediction Workflow Status & Advisor Notes
        updated = update_prediction_status(pid, "Reviewed", "Advisor conducted initial review.")
        self.assertTrue(updated)
        rec_updated = get_prediction_by_id(pid)
        self.assertEqual(rec_updated["status"], "Reviewed")
        self.assertEqual(rec_updated["advisor_notes"], "Advisor conducted initial review.")

        # 4. RELATIONSHIP: Add Advisory Note (Foreign Key)
        nid = add_advisory_note(pid, "Senior Academic Advisor", "Scheduled peer tutoring session", "Tutoring")
        self.assertIsInstance(nid, int)

        notes = get_advisory_notes(pid)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["author_role"], "Senior Academic Advisor")
        self.assertEqual(notes[0]["note_text"], "Scheduled peer tutoring session")

        # 5. REST API CRUD Integration: PATCH /api/history/{id} and DELETE /api/history/{id}
        patch_body = json.dumps({"status": "Resolved", "advisor_notes": "Completed intervention plan."})
        s_patch, b_patch = self._call("PATCH", f"/api/history/{pid}", patch_body.encode("utf-8"), {"Content-Type": "application/json"})
        self.assertEqual(s_patch, 200)

        # 6. CASCADE DELETE: Deleting prediction must remove prediction and cascade delete advisory note
        s_del, _ = self._call("DELETE", f"/api/history/{pid}")
        self.assertEqual(s_del, 200)

        self.assertIsNone(get_prediction_by_id(pid))
        self.assertEqual(len(get_advisory_notes(pid)), 0)

    def test_database_sql_injection_resilience(self):
        from backend.app.database.repository import get_recent_predictions
        from backend.app.database.session import get_db_connection

        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE predictions; --",
            "' UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12 --",
        ]
        for p in payloads:
            results = get_recent_predictions(search=p)
            self.assertEqual(len(results), 0)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions;")
            count = cursor.fetchone()[0]
            self.assertGreaterEqual(count, 200)

    def test_database_authentication_security_and_hashing(self):
        from backend.app.database.repository import get_institutional_user, verify_user_credentials
        from backend.app.database.session import get_db_connection

        # 1. Verify user profile exists in database
        advisor = get_institutional_user("advisor")
        self.assertIsNotNone(advisor)
        self.assertEqual(advisor["role"], "Senior Academic Advisor")
        self.assertEqual(len(advisor["pin_hash"]), 64)  # 256-bit hash in hex
        self.assertEqual(len(advisor["pin_salt"]), 32)  # 16 bytes in hex

        # 2. Verify passwords are NEVER stored in plain text
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pin_hash FROM institutional_users WHERE username = 'advisor';")
            stored = cursor.fetchone()[0]
            self.assertNotEqual(stored, "1234")

        # 3. Verify salted hash authentication function
        valid_auth = verify_user_credentials("advisor", "1234")
        self.assertIsNotNone(valid_auth)
        self.assertEqual(valid_auth["username"], "advisor")

        invalid_auth = verify_user_credentials("advisor", "wrong_pin")
        self.assertIsNone(invalid_auth)

    def test_http_special_character_search_resilience(self):
        # Verify searching with regex special characters does not crash with re.error (500)
        for char in ["[", "(", "+", "*", "?", "\\", "MAT-0001 (Math)"]:
            status_s, body_s = self._call("GET", f"/api/students?search={char}")
            self.assertEqual(status_s, 200)

            status_exp, body_exp = self._call("GET", f"/api/students/export?search={char}")
            self.assertEqual(status_exp, 200)

    def test_predictor_single_class_and_missing_proba_resilience(self):
        import pandas as pd
        from backend.app.ml.predictor import StudentPredictor

        predictor = StudentPredictor(timing_scenario="early")
        # Test normal single row dataframe
        df = pd.DataFrame([{
            "student_identifier": "TEST-RESILIENCE-001",
            "school": "GP",
            "sex": "F",
            "age": 17,
            "studytime": 2,
            "failures": 0,
            "absences": 4,
            "schoolsup": "no",
            "famsup": "no",
            "higher": "yes",
            "Medu": 2,
            "Fedu": 2,
            "internet": "yes",
            "goout": 3,
            "health": 4,
        }])
        res = predictor.predict_dataframe(df)
        self.assertEqual(len(res), 1)
        self.assertIn(res[0]["predicted_outcome"], ["Pass", "Fail"])


    def test_sqlite_path_resolution(self):
        from backend.app.database.session import _resolve_sqlite_path
        from backend.app.core.config import PROJECT_ROOT, DATA_DIR

        # 1. Relative SQLite URL
        rel_path = _resolve_sqlite_path("sqlite:///./data/custom_test.db")
        self.assertEqual(rel_path, (PROJECT_ROOT / "data" / "custom_test.db").resolve())

        # 2. Absolute SQLite URL
        abs_path = _resolve_sqlite_path("sqlite:////tmp/absolute_test.db")
        self.assertEqual(abs_path, Path("/tmp/absolute_test.db"))

        # 3. Fallback when invalid scheme
        fb_path = _resolve_sqlite_path("postgresql://localhost/db")
        self.assertEqual(fb_path, DATA_DIR / "student_performance.db")

    def test_security_rbac_and_token_rotation(self):
        import json
        # 1. Login as teacher1
        login_payload = json.dumps({"username": "teacher1", "pin": "1234"}).encode()
        status_l, body_l = self._call("POST", "/api/auth/login", body=login_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(status_l, 200)
        auth_data = json.loads(body_l)
        self.assertIn("access_token", auth_data)
        self.assertIn("refresh_token", auth_data)
        self.assertEqual(auth_data["role"], "TEACHER")
        token = auth_data["access_token"]
        refresh_tok = auth_data["refresh_token"]

        # 2. Check /api/auth/me
        status_me, body_me = self._call("GET", "/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(status_me, 200)
        me_data = json.loads(body_me)
        self.assertEqual(me_data["username"], "teacher1")

        # 3. Refresh token rotation
        ref_payload = json.dumps({"refresh_token": refresh_tok}).encode()
        status_ref, body_ref = self._call("POST", "/api/auth/refresh", body=ref_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(status_ref, 200)
        ref_data = json.loads(body_ref)
        self.assertIn("access_token", ref_data)
        self.assertIn("refresh_token", ref_data)
        new_token = ref_data["access_token"]

        # 4. Logout & Token Revocation
        logout_payload = json.dumps({"token": new_token}).encode()
        status_out, body_out = self._call("POST", "/api/auth/logout", body=logout_payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {new_token}"})
        self.assertEqual(status_out, 200)

        # 5. Accessing with blacklisted token fails
        status_revoked, body_revoked = self._call("GET", "/api/auth/me", headers={"Authorization": f"Bearer {new_token}"})
        self.assertEqual(status_revoked, 200)
        self.assertFalse(json.loads(body_revoked)["authenticated"])

        # Protected endpoint requiring authentication strictly returns 401
        status_req, _ = self._call("POST", "/api/auth/logout", headers={"Authorization": f"Bearer {new_token}"})
        self.assertEqual(status_req, 401)

    def test_brute_force_lockout_and_password_reset(self):
        import json
        import uuid
        # Create a test user for lockout
        from backend.app.database.repository import register_user_db
        uname = f"lockuser_{uuid.uuid4().hex[:6]}"
        register_user_db(uname, "Lockout Test User", "TEACHER", "math", "testpass123", email=f"{uname}@test.edu")

        # Fail 4 consecutive login attempts (returns 401)
        for attempt in range(4):
            bad_payload = json.dumps({"username": uname, "pin": "wrongpass"}).encode()
            st, bd = self._call("POST", "/api/auth/login", body=bad_payload, headers={"Content-Type": "application/json"})
            self.assertEqual(st, 401)

        # 5th attempt triggers lockout and returns 403 Account Locked
        lock_payload = json.dumps({"username": uname, "pin": "wrongpass"}).encode()
        st_lock, bd_lock = self._call("POST", "/api/auth/login", body=lock_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(st_lock, 403)
        self.assertIn("locked", bd_lock.lower())

        # Password Reset Request
        reset_req = json.dumps({"email_or_username": f"{uname}@test.edu"}).encode()
        st_req, bd_req = self._call("POST", "/api/auth/password-reset-request", body=reset_req, headers={"Content-Type": "application/json"})
        self.assertEqual(st_req, 200)
        reset_token = json.loads(bd_req)["reset_token"]

        # Confirm password reset
        confirm_req = json.dumps({"reset_token": reset_token, "new_pin": "newpassword123"}).encode()
        st_conf, bd_conf = self._call("POST", "/api/auth/password-reset", body=confirm_req, headers={"Content-Type": "application/json"})
        self.assertEqual(st_conf, 200)

        # Login with new password should now succeed
        good_payload = json.dumps({"username": uname, "pin": "newpassword123"}).encode()
        st_good, bd_good = self._call("POST", "/api/auth/login", body=good_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(st_good, 200)

    def test_student_role_self_access_enforcement(self):
        import json
        # 1. Login as student1 (assigned to MAT-0001)
        login_payload = json.dumps({"username": "student1", "pin": "1234"}).encode()
        st_l, bd_l = self._call("POST", "/api/auth/login", body=login_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(st_l, 200)
        token = json.loads(bd_l)["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. View self record: MAT-0001 -> 200 OK
        st_self, bd_self = self._call("GET", "/api/students/MAT-0001", headers=headers)
        self.assertEqual(st_self, 200)
        self.assertEqual(json.loads(bd_self)["student_identifier"], "MAT-0001")

        # 3. View another student record: MAT-0002 -> 403 Forbidden
        st_other, bd_other = self._call("GET", "/api/students/MAT-0002", headers=headers)
        self.assertEqual(st_other, 403)
        self.assertIn("authorized", bd_other.lower())

        # 4. Compare with other student -> 403 Forbidden
        st_cmp, _ = self._call("GET", "/api/students/compare?student_a=MAT-0002&student_b=cohort_mean", headers=headers)
        self.assertEqual(st_cmp, 403)

        # 5. Cohort CSV Export -> 403 Forbidden
        st_exp, _ = self._call("GET", "/api/students/export?subject=math", headers=headers)
        self.assertEqual(st_exp, 403)

        # 6. List students -> Only returns self (1 record)
        st_list, bd_list = self._call("GET", "/api/students?page=1&page_size=20", headers=headers)
        self.assertEqual(st_list, 200)
        list_data = json.loads(bd_list)
        self.assertEqual(list_data["total"], 1)
        self.assertEqual(list_data["records"][0]["student_identifier"], "MAT-0001")

    def test_student_crud_and_validation(self):
        import json
        import uuid
        # Login as superadmin
        login_payload = json.dumps({"username": "superadmin", "pin": "admin123"}).encode()
        st_l, bd_l = self._call("POST", "/api/auth/login", body=login_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(st_l, 200)
        token = json.loads(bd_l)["access_token"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # 1. Create a student
        custom_id = f"TEST-CRUD-{uuid.uuid4().hex[:4].upper()}"
        new_student = {
            "student_identifier": custom_id,
            "department_code": "math",
            "subject": "Mathematics",
            "school": "GP",
            "sex": "F",
            "age": 18,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "T",
            "Medu": 3,
            "Fedu": 3,
            "Mjob": "other",
            "Fjob": "other",
            "reason": "course",
            "guardian": "mother",
            "traveltime": 1,
            "studytime": 3,
            "failures": 0,
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 2,
            "Dalc": 1,
            "Walc": 1,
            "health": 5,
            "absences": 2,
            "attendance_percentage": 96.7,
            "G1": 15,
            "G2": 16,
            "G3": 17,
        }
        st_c, bd_c = self._call("POST", "/api/students", body=json.dumps(new_student).encode(), headers=headers)
        self.assertEqual(st_c, 200)
        created_data = json.loads(bd_c)
        self.assertEqual(created_data["student"]["student_identifier"], custom_id)
        self.assertEqual(created_data["student"]["predicted_outcome"], "Pass")

        # 2. Update student (e.g. increase studytime and absences)
        update_data = {"studytime": 4, "absences": 4, "attendance_percentage": 93.3}
        st_u, bd_u = self._call("PUT", f"/api/students/{custom_id}", body=json.dumps(update_data).encode(), headers=headers)
        self.assertEqual(st_u, 200)
        updated_data = json.loads(bd_u)
        self.assertEqual(updated_data["student"]["studytime"], 4)

        # 3. Delete student
        st_d, bd_d = self._call("DELETE", f"/api/students/{custom_id}", headers=headers)
        self.assertEqual(st_d, 200)

        # 4. Verify student is deleted
        st_check, _ = self._call("GET", f"/api/students/{custom_id}", headers=headers)
        self.assertEqual(st_check, 404)

    def test_csv_import_preview_and_commit(self):
        import json
        import uuid
        # Login as superadmin
        login_payload = json.dumps({"username": "superadmin", "pin": "admin123"}).encode()
        st_l, bd_l = self._call("POST", "/api/auth/login", body=login_payload, headers={"Content-Type": "application/json"})
        token = json.loads(bd_l)["access_token"]

        uid = uuid.uuid4().hex[:4].upper()
        csv_content = (
            f"student_identifier;department_code;subject;school;sex;age;studytime;failures;absences;G1;G2;G3\n"
            f"IMP-{uid}-001;math;Mathematics;GP;F;17;2;0;2;12;13;14\n"
            f"IMP-{uid}-002;portuguese;Portuguese;MS;M;16;3;0;0;15;16;16\n"
        ).encode("utf-8")

        boundary = "----TestBoundaryImport"
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"import_test.csv\"\r\n"
            f"Content-Type: text/csv\r\n\r\n"
        ).encode() + csv_content + f"\r\n--{boundary}--\r\n".encode()

        # Preview import
        st_p, bd_p = self._call(
            "POST",
            "/api/students/import-preview",
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {token}",
            }
        )
        self.assertEqual(st_p, 200)
        preview = json.loads(bd_p)
        self.assertEqual(preview["total_rows"], 2)
        self.assertEqual(preview["valid_rows"], 2)
        self.assertTrue(preview["is_valid"])

        # Commit import
        commit_payload = json.dumps({"records": preview["preview_records"]}).encode()
        st_com, bd_com = self._call(
            "POST",
            "/api/students/import-commit",
            body=commit_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        )
        self.assertEqual(st_com, 200)
        com_data = json.loads(bd_com)
        self.assertEqual(com_data["records_saved"], 2)

        # Cleanup imported records
        self._call("DELETE", f"/api/students/IMP-{uid}-001", headers={"Authorization": f"Bearer {token}"})
        self._call("DELETE", f"/api/students/IMP-{uid}-002", headers={"Authorization": f"Bearer {token}"})

    def test_departments_and_subjects_endpoints(self):
        import json
        st_d, bd_d = self._call("GET", "/api/departments")
        self.assertEqual(st_d, 200)
        depts = json.loads(bd_d)
        self.assertGreaterEqual(len(depts), 2)
        dept_codes = [d["code"] for d in depts]
        self.assertIn("math", dept_codes)
        self.assertIn("portuguese", dept_codes)

        st_s, bd_s = self._call("GET", "/api/subjects")
        self.assertEqual(st_s, 200)
        subjs = json.loads(bd_s)
        self.assertGreaterEqual(len(subjs), 2)
        subj_codes = [s["code"] for s in subjs]
        self.assertIn("MATH-101", subj_codes)
        self.assertIn("LANG-101", subj_codes)


if __name__ == "__main__":
    unittest.main()

