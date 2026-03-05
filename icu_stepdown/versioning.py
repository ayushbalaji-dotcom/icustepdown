import json
from typing import Any, Dict, List

from .quality import sha256_of_text


def schema_hash(feature_list: List[str]) -> str:
    text = json.dumps(feature_list, separators=(",", ":"))
    return sha256_of_text(text)


def config_hash(cfg: Dict[str, Any]) -> str:
    text = json.dumps(cfg, sort_keys=True, default=str)
    return sha256_of_text(text)


