import cgi
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from datetime import datetime

import pandas as pd

from .config import load_config
from .features import compute_features
from .model_store import resolve_runtime_model_path, runtime_model_status
from .patient_store import append_row, load_preop, load_rows, save_preop, start_encounter
from .preprocess import preprocess
from .quality import QualityLogger
from .schema import validate_raw
from .score import score_features, _fail_closed_dashboard
from .training_service import train_workbook


class StepdownHandler(BaseHTTPRequestHandler):
    server_version = "ICUStepdown/0.1"

    @staticmethod
    def _parse_timestamp(value: str) -> str:
        # Accept common formats and normalize to ISO8601
        ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            raise ValueError("invalid_timestamp_format")
        return ts.isoformat()

    @staticmethod
    def _parse_optional_float(value):
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        try:
            return float(text.replace(",", ""))
        except ValueError as exc:
            raise ValueError("invalid_number") from exc

    @staticmethod
    def _parse_yes_no(value):
        if value in (True, 1, "1", "Yes", "yes", "Y", "y"):
            return 1.0
        if value in (False, 0, "0", "No", "no", "N", "n"):
            return 0.0
        return None

    @staticmethod
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: StepdownHandler._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [StepdownHandler._sanitize(v) for v in obj]
        if pd.isna(obj):
            return None
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return obj

    @staticmethod
    def _json_default(obj):
        if pd.isna(obj):
            return None
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return str(obj)

    def _send_json(self, code: int, payload: dict):
        clean = self._sanitize(payload)
        body = json.dumps(clean, default=self._json_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_file(os.path.join(self.server.static_dir, "index.html"), "text/html")
        if parsed.path == "/app.js":
            return self._send_file(os.path.join(self.server.static_dir, "app.js"), "application/javascript")
        if parsed.path == "/styles.css":
            return self._send_file(os.path.join(self.server.static_dir, "styles.css"), "text/css")
        if parsed.path == "/api/preop":
            qs = parse_qs(parsed.query)
            nhs = qs.get("nhs_number", [""])[0]
            return self._handle_preop_get(nhs)
        if parsed.path == "/api/score":
            qs = parse_qs(parsed.query)
            nhs = qs.get("nhs_number", [""])[0]
            return self._handle_score(nhs)
        if parsed.path == "/api/model-status":
            return self._handle_model_status()
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/train":
            return self._handle_train()

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "invalid_json"})

        if self.path == "/api/start":
            nhs = payload.get("nhs_number")
            if not nhs:
                return self._send_json(400, {"error": "missing_nhs_number"})
            try:
                encounter_id = start_encounter(self.server.db_path, nhs, force_new=bool(payload.get("force_new")))
                return self._send_json(200, {"status": "ok", "encounter_id": encounter_id})
            except Exception as e:
                return self._send_json(400, {"error": str(e)})

        if self.path == "/api/preop":
            nhs = payload.get("nhs_number")
            preop = payload.get("preop", {})
            if not nhs:
                return self._send_json(400, {"error": "missing_nhs_number"})
            try:
                normalized = {
                    "age_years": self._parse_optional_float(preop.get("age_years")),
                    "bmi": self._parse_optional_float(preop.get("bmi")),
                    "frailty_score": self._parse_optional_float(preop.get("frailty_score")),
                    "renal_function": self._parse_optional_float(preop.get("renal_function")),
                    "lv_function": self._parse_optional_float(preop.get("lv_function")),
                    "diabetes": self._parse_yes_no(preop.get("diabetes")),
                }
                encounter_id = save_preop(self.server.db_path, nhs, normalized)
                return self._send_json(200, {"status": "ok", "encounter_id": encounter_id})
            except Exception as e:
                return self._send_json(400, {"error": str(e)})

        if self.path == "/api/append":
            nhs = payload.get("nhs_number")
            row = payload.get("row", {})
            if not nhs:
                return self._send_json(400, {"error": "missing_nhs_number"})
            if not row:
                return self._send_json(400, {"error": "missing_row"})
            if not row.get("timestamp"):
                return self._send_json(400, {"error": "missing_timestamp"})
            try:
                row["timestamp"] = self._parse_timestamp(row["timestamp"])
                append_row(self.server.db_path, nhs, row)
                return self._send_json(200, {"status": "ok"})
            except Exception as e:
                return self._send_json(400, {"error": str(e)})

        self.send_response(404)
        self.end_headers()

    def _handle_preop_get(self, nhs_number: str):
        if not nhs_number:
            return self._send_json(400, {"error": "missing_nhs_number"})
        try:
            preop = load_preop(self.server.db_path, nhs_number)
        except Exception as e:
            return self._send_json(400, {"error": str(e)})
        if not preop:
            return self._send_json(200, {"status": "no_data"})
        return self._send_json(200, {"status": "ok", "preop": preop})

    def _handle_model_status(self):
        model_state = runtime_model_status(self.server.model_path, self.server.base_dir)
        status = "ok" if model_state.get("available") else "no_model"
        return self._send_json(200, {"status": status, "model": model_state})

    def _handle_train(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return self._send_json(400, {"error": "expected_multipart_form_data"})

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "training_file" not in form:
            return self._send_json(400, {"error": "missing_training_file"})

        file_item = form["training_file"]
        filename = os.path.basename(getattr(file_item, "filename", "") or "training.xlsx")
        if not filename.lower().endswith(".xlsx"):
            return self._send_json(400, {"error": "training_file_must_be_xlsx"})
        model_name = (form.getfirst("model_name", "") or "").strip() or None

        fd, temp_path = tempfile.mkstemp(suffix=".xlsx", dir=self.server.base_dir)
        os.close(fd)
        try:
            with open(temp_path, "wb") as f:
                f.write(file_item.file.read())
            result = train_workbook(
                temp_path,
                config_path=self.server.config_path,
                base_dir=self.server.base_dir,
                model_name=model_name or os.path.splitext(filename)[0],
                activate=True,
            )
            self.server.model_path = result["model_path"]
            return self._send_json(
                200,
                {
                    "status": "ok",
                    "summary": result["summary"],
                    "model": runtime_model_status(self.server.model_path, self.server.base_dir),
                },
            )
        except Exception as e:
            return self._send_json(400, {"error": str(e)})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _handle_score(self, nhs_number: str):
        if not nhs_number:
            return self._send_json(400, {"error": "missing_nhs_number"})

        ql = QualityLogger()
        rows = load_rows(self.server.db_path, nhs_number)
        if not rows:
            return self._send_json(200, {"status": "no_data"})
        raw = pd.DataFrame(rows)

        try:
            raw, _ = validate_raw(raw, self.server.cfg, ql)
            raw = preprocess(raw, self.server.cfg, ql)
        except Exception as e:
            return self._send_json(200, {"status": "error", "message": str(e)})

        t0 = pd.to_datetime(raw["timestamp"]).min()
        t1 = pd.to_datetime(raw["timestamp"]).max()
        hours_span = 0 if pd.isna(t0) or pd.isna(t1) else (t1 - t0).total_seconds() / 3600.0
        if pd.isna(t0) or pd.isna(t1) or hours_span < 4:
            return self._send_json(200, {
                "status": "insufficient_data",
                "hours": hours_span,
                "min_timestamp": None if pd.isna(t0) else str(t0),
                "max_timestamp": None if pd.isna(t1) else str(t1),
                "row_count": int(len(raw)),
            })

        features = compute_features(raw, self.server.cfg, ql)
        if features.empty:
            return self._send_json(200, {"status": "insufficient_data"})

        model_bundle = None
        model_state = runtime_model_status(self.server.model_path, self.server.base_dir)
        runtime_model_path = resolve_runtime_model_path(self.server.model_path, self.server.base_dir)
        warning = None
        if self.server.use_baseline:
            from .baseline import build_baseline_bundle

            model_bundle = build_baseline_bundle(features, self.server.cfg)
            scores, _, dashboard, _ = score_features(features, model_bundle, self.server.cfg, ql, force_schema=True)
            warning = "baseline_model_used"
        else:
            if runtime_model_path and os.path.exists(runtime_model_path):
                try:
                    from .train import load_model_bundle

                    model_bundle = load_model_bundle(runtime_model_path)
                except Exception:
                    model_bundle = None
            if model_bundle is None:
                scores, _, dashboard, _ = _fail_closed_dashboard(features, ql)
                warning = "no_model_loaded_fail_closed"
            else:
                scores, _, dashboard, _ = score_features(features, model_bundle, self.server.cfg, ql, force_schema=True)

        latest = dashboard.iloc[0].to_dict() if not dashboard.empty else {}
        domain = {
            "Respiratory": latest.get("limiting_factor") == "Respiratory",
            "Neurological": latest.get("limiting_factor") == "Neuro-Infection",
            "Cardiovascular": latest.get("limiting_factor") == "Haemodynamic",
            "Surgical": latest.get("limiting_factor") == "Bleeding",
            "Haemodynamics": latest.get("limiting_factor") == "Haemodynamic",
            "Imaging": "Not assessed by model",
        }

        return self._send_json(200, {
            "status": "ok",
            "dashboard": latest,
            "domain_flags": domain,
            "warning": warning,
            "model_status": model_state,
        })

    def log_message(self, format, *args):
        # Avoid logging request paths that might contain identifiers.
        return


class StepdownServer(HTTPServer):
    def __init__(self, host: str, port: int, static_dir: str, db_path: str, config_path: str, model_path: str | None, use_baseline: bool):
        super().__init__((host, port), StepdownHandler)
        self.static_dir = static_dir
        self.db_path = db_path
        self.base_dir = os.path.dirname(os.path.abspath(db_path)) or os.path.abspath("database")
        self.cfg = load_config(config_path)
        self.config_path = config_path
        self.model_path = model_path
        self.use_baseline = bool(use_baseline)
