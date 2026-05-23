"""
CAGE 4 Action Space Definition
-------------------------------
Defines valid Blue agent actions for CybORG CAGE 4.
Includes regex + fuzzy parsing of LLM outputs.

NOTE: We attack AGAINST LLM defenders (not WITH LLMs against RL defenders).
This framing is critical — see Castro et al. 2505.04843 for the LLM ACD baseline.
"""

import re
import difflib

# All valid Blue agent actions in CAGE 4
BLUE_ACTIONS = [
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "DeployDecoy",
    "BlockTraffic",
    "AllowTraffic",
]

# High-consequence actions requiring extra caution / consistency guard
CRITICAL_ACTIONS = ["Remove", "Restore", "BlockTraffic"]

# Regex patterns for extracting action names from LLM output
_ACTION_PATTERNS = [
    # Explicit "ACTION: <name>" format
    re.compile(r"ACTION:\s*([A-Za-z]+)", re.IGNORECASE),
    # "action is <name>" or "action: <name>"
    re.compile(r"action[:\s]+is[:\s]+([A-Za-z]+)", re.IGNORECASE),
    # "recommend <name>" or "recommending <name>"
    re.compile(r"recommend(?:ing)?\s+([A-Za-z]+)", re.IGNORECASE),
    # "execute <name>"
    re.compile(r"execute[:\s]+([A-Za-z]+)", re.IGNORECASE),
    # "selected action: <name>"
    re.compile(r"selected\s+action[:\s]+([A-Za-z]+)", re.IGNORECASE),
    # standalone capitalized word that matches an action
    re.compile(r"\b(" + "|".join(BLUE_ACTIONS) + r")\b"),
]


def parse_llm_output(text: str) -> str:
    """
    Extract a valid CAGE 4 Blue action from raw LLM output text.

    Strategy:
    1. Try each regex pattern in priority order.
    2. If regex fails, use fuzzy matching against BLUE_ACTIONS.
    3. Default to "Monitor" if no valid action found.

    Args:
        text: Raw LLM output string (may contain reasoning + action).

    Returns:
        A valid action string from BLUE_ACTIONS.
    """
    if not text or not text.strip():
        return "Monitor"

    # Normalize whitespace
    text_clean = " ".join(text.split())

    # Try each pattern
    for pattern in _ACTION_PATTERNS:
        match = pattern.search(text_clean)
        if match:
            candidate = match.group(1)
            # Case-insensitive lookup in BLUE_ACTIONS
            candidate_lower = candidate.lower()
            for action in BLUE_ACTIONS:
                if action.lower() == candidate_lower:
                    return action

    # Fuzzy fallback: find closest match
    text_words = text_clean.split()
    best_action = None
    best_score = 0.0
    for word in text_words:
        matches = difflib.get_close_matches(word, BLUE_ACTIONS, n=1, cutoff=0.7)
        if matches:
            score = difflib.SequenceMatcher(None, word.lower(), matches[0].lower()).ratio()
            if score > best_score:
                best_score = score
                best_action = matches[0]

    if best_action:
        return best_action

    return "Monitor"


def get_action_prompt_description() -> str:
    """Return a human-readable description of available actions for LLM prompting."""
    return (
        "Available defensive actions:\n"
        "  - Monitor:      Observe host for anomalies (low cost, passive)\n"
        "  - Analyse:      Deep forensic analysis of a host\n"
        "  - Remove:       Remove malware/threat from a host [CRITICAL]\n"
        "  - Restore:      Restore host to clean snapshot [CRITICAL]\n"
        "  - DeployDecoy:  Deploy honeypot to attract red agent\n"
        "  - BlockTraffic: Block network traffic to/from a host [CRITICAL]\n"
        "  - AllowTraffic: Re-allow traffic to a previously blocked host\n"
    )
