import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def require_keys(cfg: Dict[str, Any], keys: list) -> None:
    for k in keys:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")


