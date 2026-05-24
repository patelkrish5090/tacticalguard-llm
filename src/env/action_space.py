"""Blue-agent action space and parser for LLM responses."""

from __future__ import annotations

import re
from difflib import get_close_matches


BLUE_ACTIONS = [
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "DeployDecoy",
    "BlockTraffic",
    "AllowTraffic",
]

CRITICAL_ACTIONS = ["Remove", "Restore", "BlockTraffic"]

_ACTION_BY_NORMALIZED = {
    re.sub(r"[^a-z]", "", action.lower()): action for action in BLUE_ACTIONS
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def parse_llm_output(text: str | None) -> str:
    """Extract the first valid blue action from an LLM response."""
    if not text:
        return "Monitor"

    # Prefer explicit ACTION lines.
    action_line = re.search(
        r"^\s*ACTION\s*[:\-]\s*([A-Za-z _-]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    candidates = []
    if action_line:
        candidates.append(action_line.group(1))
    candidates.extend(re.findall(r"[A-Za-z][A-Za-z _-]*", text))

    for candidate in candidates:
        normalized = _normalize(candidate)
        for key, action in _ACTION_BY_NORMALIZED.items():
            if key in normalized:
                return action

        match = get_close_matches(normalized, _ACTION_BY_NORMALIZED.keys(), n=1, cutoff=0.78)
        if match:
            return _ACTION_BY_NORMALIZED[match[0]]

    return "Monitor"
