import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import typer

from .config import load_config
from .features import compute_features
from .io_excel import read_excel_sheets, write_excel_preserve
from .labels import build_labels
from .preprocess import preprocess
from .quality import QualityLogger, sha256_of_file
from .schema import validate_outcomes, validate_raw
from .score import score_features
from .versioning import config_hash, schema_hash

app = typer.Typer(add_completion=False)


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _run_manifest(
    cfg: Dict[str, Any],
    feature_list: list,
    input_path: str,
    output_path: Optional[str],
    model_bundle: Optional[Dict[str, Any]],
    scores: Optional[pd.DataFrame],
    calibration_method: Optional[str],
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = {
        "tool_version": "0.1.0",
        "git_commit": _git_commit(),
        "model_bundle_hash": None,
        "config_hash": config_hash(cfg),
        "feature_schema_hash": schema_hash(feature_list),
        "feature_list": feature_list,
        "input_file_hash": sha256_of_file(input_path),
        "output_file_hash": None,
        "timestamp": datetime.utcnow().isoformat(),
        "counts": {},
        "calibration_method": calibration_method,
    }
    if model_path and os.path.exists(model_path):
        manifest["model_bundle_hash"] = sha256_of_file(model_path)
    elif model_bundle is not None:
        manifest["model_bundle_hash"] = schema_hash(model_bundle.get("feature_columns", []))
    if scores is not None:
        counts = scores["traffic_light"].value_counts().to_dict() if "traffic_light" in scores else {}
        manifest["counts"] = counts
        manifest["data_quality_capped"] = int((scores["traffic_light"] == "AMBER").sum()) if "traffic_light" in scores else 0
    if output_path and os.path.exists(output_path):
        manifest["output_file_hash"] = sha256_of_file(output_path)
    return manifest


@app.command()
def validate(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    debug: bool = typer.Option(False, "--debug"),
):
    try:
        cfg = load_config(config)
        ql = QualityLogger()
        sheets = read_excel_sheets(input, ["raw_icu_data"])
        if "raw_icu_data" not in sheets:
            raise ValueError("Missing required sheet: raw_icu_data")
        raw = sheets["raw_icu_data"]
        raw, _ = validate_raw(raw, cfg, ql)
        raw = preprocess(raw, cfg, ql)

        ql.add("INFO", "rows_count", total=int(len(raw)))
        ql.add("INFO", "rejected_rows_count", total=int(len(ql.rejected_rows)))

        out_sheets = {
            "features_4h": pd.DataFrame(),
            "scores_4h": pd.DataFrame(),
            "signals_explained": pd.DataFrame(),
            "dashboard": pd.DataFrame(),
            "quality_log": ql.to_dataframe(),
            "rejected_rows": ql.rejected_to_dataframe(),
        }
        write_excel_preserve(input, output, out_sheets)
        manifest = _run_manifest(cfg, cfg["feature_schema"], input, output, None, None, None)
        with open(os.path.splitext(output)[0] + ".run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        write_excel_preserve(input, output, {"run_manifest": pd.DataFrame([manifest])})
    except Exception as e:
        if debug:
            raise
        typer.echo(str(e), err=True)
        sys.exit(1)


@app.command()
def features(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    debug: bool = typer.Option(False, "--debug"),
):
    try:
        cfg = load_config(config)
        ql = QualityLogger()
        sheets = read_excel_sheets(input, ["raw_icu_data"])
        if "raw_icu_data" not in sheets:
            raise ValueError("Missing required sheet: raw_icu_data")
        raw = sheets["raw_icu_data"]
        raw, _ = validate_raw(raw, cfg, ql)
        raw = preprocess(raw, cfg, ql)
        feat = compute_features(raw, cfg, ql)

        ql.add("INFO", "rows_count", total=int(len(raw)))
        ql.add("INFO", "rejected_rows_count", total=int(len(ql.rejected_rows)))

        out_sheets = {
            "features_4h": feat,
            "scores_4h": pd.DataFrame(),
            "signals_explained": pd.DataFrame(),
            "dashboard": pd.DataFrame(),
            "quality_log": ql.to_dataframe(),
            "rejected_rows": ql.rejected_to_dataframe(),
        }
        write_excel_preserve(input, output, out_sheets)
        manifest = _run_manifest(cfg, cfg["feature_schema"], input, output, None, None, None)
        with open(os.path.splitext(output)[0] + ".run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        write_excel_preserve(input, output, {"run_manifest": pd.DataFrame([manifest])})
    except Exception as e:
        if debug:
            raise
        typer.echo(str(e), err=True)
        sys.exit(1)


@app.command()
def train(
    input: str = typer.Option(..., "--input"),
    model_out: str = typer.Option(..., "--model-out"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    debug: bool = typer.Option(False, "--debug"),
):
    try:
        cfg = load_config(config)
        ql = QualityLogger()
        sheets = read_excel_sheets(input, ["features_4h", "outcomes"])
        if "features_4h" not in sheets:
            raise ValueError("Missing required sheet: features_4h")
        if "outcomes" not in sheets:
            raise ValueError("Missing required sheet: outcomes")
        features_df = sheets["features_4h"]
        outcomes = sheets["outcomes"]
        validate_outcomes(outcomes, cfg)
        outcomes = build_labels(outcomes, cfg, ql)

        from .train import save_model_bundle, train_model
        bundle, metrics = train_model(features_df, outcomes, cfg, ql)
        save_model_bundle(bundle, model_out)

        ql.add("INFO", "training_complete", metrics=metrics)
    except Exception as e:
        if debug:
            raise
        typer.echo(str(e), err=True)
        sys.exit(1)


@app.command()
def score(
    input: str = typer.Option(..., "--input"),
    model_in: Optional[str] = typer.Option(None, "--model-in"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    debug: bool = typer.Option(False, "--debug"),
    force_schema: bool = typer.Option(False, "--force-schema"),
    baseline: bool = typer.Option(False, "--baseline"),
):
    try:
        cfg = load_config(config)
        ql = QualityLogger()
        sheets = read_excel_sheets(input, ["features_4h"])
        if "features_4h" not in sheets:
            raise ValueError("Missing required sheet: features_4h")
        features_df = sheets["features_4h"]
        if baseline:
            from .baseline import build_baseline_bundle

            model_bundle = build_baseline_bundle(features_df, cfg)
            scores, signals, dashboard, _ = score_features(features_df, model_bundle, cfg, ql, force_schema=True)
        else:
            if not model_in:
                raise ValueError("Missing required --model-in (or use --baseline).")
            try:
                from .train import load_model_bundle

                model_bundle = load_model_bundle(model_in)
                scores, signals, dashboard, _ = score_features(features_df, model_bundle, cfg, ql, force_schema=force_schema)
            except Exception:
                from .score import _fail_closed_dashboard

                scores, signals, dashboard, _ = _fail_closed_dashboard(features_df, ql)
                model_bundle = None

        out_sheets = {
            "scores_4h": scores,
            "signals_explained": signals,
            "dashboard": dashboard,
            "quality_log": ql.to_dataframe(),
            "rejected_rows": ql.rejected_to_dataframe(),
        }
        write_excel_preserve(input, output, out_sheets)
        calibration_method = model_bundle.get("metrics", {}).get("calibration_method") if model_bundle else None
        manifest = _run_manifest(cfg, cfg["feature_schema"], input, output, model_bundle, scores, calibration_method, model_path=model_in)
        with open(os.path.splitext(output)[0] + ".run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        write_excel_preserve(input, output, {"run_manifest": pd.DataFrame([manifest])})
    except Exception as e:
        if debug:
            raise
        typer.echo(str(e), err=True)
        sys.exit(1)


@app.command(name="run-all")
def run_all(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    model_out: str = typer.Option(..., "--model-out"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    debug: bool = typer.Option(False, "--debug"),
    force_schema: bool = typer.Option(False, "--force-schema"),
):
    try:
        cfg = load_config(config)
        ql = QualityLogger()
        sheets = read_excel_sheets(input, ["raw_icu_data", "outcomes"])
        if "raw_icu_data" not in sheets:
            raise ValueError("Missing required sheet: raw_icu_data")
        raw = sheets["raw_icu_data"]
        raw, _ = validate_raw(raw, cfg, ql)
        raw = preprocess(raw, cfg, ql)
        feat = compute_features(raw, cfg, ql)

        if "outcomes" not in sheets:
            raise ValueError("Missing required sheet: outcomes for training")
        outcomes = sheets["outcomes"]
        validate_outcomes(outcomes, cfg)
        outcomes = build_labels(outcomes, cfg, ql)
        from .train import save_model_bundle, train_model
        bundle, _ = train_model(feat, outcomes, cfg, ql)
        save_model_bundle(bundle, model_out)

        scores, signals, dashboard, _ = score_features(feat, bundle, cfg, ql, force_schema=force_schema)

        out_sheets = {
            "features_4h": feat,
            "scores_4h": scores,
            "signals_explained": signals,
            "dashboard": dashboard,
            "quality_log": ql.to_dataframe(),
            "rejected_rows": ql.rejected_to_dataframe(),
        }
        write_excel_preserve(input, output, out_sheets)
        manifest = _run_manifest(cfg, cfg["feature_schema"], input, output, bundle, scores, bundle.get("metrics", {}).get("calibration_method"), model_path=model_out)
        with open(os.path.splitext(output)[0] + ".run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        write_excel_preserve(input, output, {"run_manifest": pd.DataFrame([manifest])})
    except Exception as e:
        if debug:
            raise
        typer.echo(str(e), err=True)
        sys.exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    model: str = typer.Option(None, "--model"),
    db_path: str = typer.Option("database/icu_stepdown.sqlite", "--db-path"),
    static_dir: str = typer.Option("frontend", "--static-dir"),
    baseline: bool = typer.Option(False, "--baseline"),
):
    from .webapp import StepdownServer

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    server = StepdownServer(host, port, static_dir, db_path, config, model, baseline)
    typer.echo(f"Serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    app()
