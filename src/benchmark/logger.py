"""
Structured logger for TacticalGuard-LLM experiments.

Writes JSONL logs with all fields needed for metric computation.
Each step record includes: episode, step, agent_id, obs, prompt, action,
reward, timing, attack metadata, and defense metadata.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Writes step-level logs to a JSONL file and keeps an in-memory buffer.
    """

    def __init__(self, log_path: str, config_name: str = ""):
        self.log_path = log_path
        self.config_name = config_name
        self._buffer: list[dict] = []

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        logger.info(f"[Logger] Writing to {log_path}")

    def log_step(self, record: dict) -> None:
        """Append a step record to the JSONL log."""
        self._buffer.append(record)
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def get_all_logs(self) -> list[dict]:
        return list(self._buffer)

    def close(self) -> None:
        self._file.close()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass


def make_log_path(output_dir: str, config_name: str) -> str:
    """Generate a timestamped log path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(output_dir, f"logs_{config_name}_{ts}.jsonl")


def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    """Save records to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    logger.info(f"[Logger] Saved {len(records)} records to {path}")
