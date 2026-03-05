import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class QualityEntry:
    time: str
    severity: str
    encounter_id: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class QualityLogger:
    def __init__(self) -> None:
        self.entries: List[QualityEntry] = []
        self.rejected_rows: List[Dict[str, Any]] = []

    def add(self, severity: str, message: str, encounter_id: Optional[str] = None, **details: Any) -> None:
        entry = QualityEntry(
            time=datetime.utcnow().isoformat(),
            severity=severity,
            encounter_id=encounter_id,
            message=message,
            details=details or {},
        )
        self.entries.append(entry)

    def reject_row(self, row: Dict[str, Any], reason: str) -> None:
        out = dict(row)
        out["reason"] = reason
        self.rejected_rows.append(out)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "time": e.time,
                "severity": e.severity,
                "encounter_id": e.encounter_id,
                "message": e.message,
                "details": json.dumps(e.details, default=str),
            }
            for e in self.entries
        ])

    def rejected_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rejected_rows)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


