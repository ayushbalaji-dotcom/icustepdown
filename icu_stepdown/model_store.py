import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def models_dir(base_dir: str = "database") -> str:
    return os.path.join(base_dir, "models")


def registry_path(base_dir: str = "database") -> str:
    return os.path.join(base_dir, "active_model.json")


def _base_record(base_dir: str) -> Dict[str, Any]:
    return {
        "available": False,
        "source": "none",
        "model_path": None,
        "registry_path": os.path.abspath(registry_path(base_dir)),
    }


def active_model_record(base_dir: str = "database") -> Dict[str, Any]:
    record = _base_record(base_dir)
    path = registry_path(base_dir)
    if not os.path.exists(path):
        return record

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        record["error"] = f"invalid_registry: {exc}"
        return record

    record.update(payload or {})
    model_path = record.get("model_path")
    if model_path:
        record["model_path"] = os.path.abspath(model_path)
        record["available"] = os.path.exists(record["model_path"])
    return record


def set_active_model(
    model_path: str,
    *,
    base_dir: str = "database",
    metrics: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(registry_path(base_dir)) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "model_path": os.path.abspath(model_path),
        "updated_at": datetime.utcnow().isoformat(),
        "metrics": metrics or {},
    }
    if metadata:
        payload.update(metadata)

    with open(registry_path(base_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return active_model_record(base_dir)


def resolve_runtime_model_path(explicit_path: Optional[str] = None, base_dir: str = "database") -> Optional[str]:
    if explicit_path:
        candidate = os.path.abspath(explicit_path)
        if os.path.exists(candidate):
            return candidate

    record = active_model_record(base_dir)
    model_path = record.get("model_path")
    if model_path and os.path.exists(model_path):
        return model_path
    return None


def runtime_model_status(explicit_path: Optional[str] = None, base_dir: str = "database") -> Dict[str, Any]:
    if explicit_path:
        candidate = os.path.abspath(explicit_path)
        if os.path.exists(candidate):
            return {
                "available": True,
                "source": "explicit",
                "model_path": candidate,
                "registry_path": os.path.abspath(registry_path(base_dir)),
            }

    record = active_model_record(base_dir)
    if record.get("available"):
        record["source"] = "active"
    elif explicit_path:
        record["source"] = "missing_explicit_fallback"
        record["requested_path"] = os.path.abspath(explicit_path)
    return record
