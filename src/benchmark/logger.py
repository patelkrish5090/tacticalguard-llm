"""
JSONL Step Logger
-----------------
Structured logger for recording every step in an episode.
Writes to JSONL format for easy analysis.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class StepLogger:
    """
    Logs every step of an episode to a JSONL file.

    Each line is a JSON object with all step-level fields.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
        self._buffer = []

    def log_step(self, step_data: Dict[str, Any]):
        """Add a step record to the buffer."""
        step_data["_timestamp"] = datetime.utcnow().isoformat()
        self._buffer.append(step_data)

    def flush(self, append: bool = True):
        """Write buffered records to disk."""
        mode = "a" if append else "w"
        with open(self.log_path, mode) as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")
        self._buffer = []

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_jsonl(path: str) -> list:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def save_jsonl(records: list, path: str, append: bool = False):
    """Save a list of records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
